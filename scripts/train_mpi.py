#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import logging
import mlflow
import os
import pickle
import psutil
import re
import socket
import subprocess
import sys
import time

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    rank = 0
    world_size = 1
    MPI_AVAILABLE = False
    comm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gpu_count():
    try:
        return len(
            [
                f
                for f in os.listdir("/dev")
                if f.startswith("nvidia") and f[6:].isdigit()
            ]
        )
    except (OSError, FileNotFoundError):
        return 0


def get_ip_from_host(host):
    ip_wait_time = 200
    counter = 0
    ip = ""
    while counter < ip_wait_time and ip == "":
        try:
            ip = socket.gethostbyname(host)
            break
        except Exception:
            counter += 1
            time.sleep(5)
    if counter == ip_wait_time and ip == "":
        raise Exception(
            "Exceeded max wait time of %ss for hostname resolution" % ip_wait_time
        )
    logger.info("IP address for %s is %s", host, ip)
    return ip


def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def parse_dcgm_output(output):
    parsed = {
        "dcgm_version": None,
        "driver_version": None,
        "gpu_device_ids": None,
        "test_results": {},
        "overall_status": "Unknown",
    }
    lines = output.split("\n")
    current_section = None
    for line in lines:
        line = line.strip()
        if "DCGM Version" in line:
            parsed["dcgm_version"] = line.split("|")[2].strip()
        elif "Driver Version Detected" in line:
            parsed["driver_version"] = line.split("|")[2].strip()
        elif "GPU Device IDs Detected" in line:
            parsed["gpu_device_ids"] = line.split("|")[2].strip()
        elif "-----  Deployment  --------" in line:
            current_section = "deployment"
        elif "-----  Integration  -------" in line:
            current_section = "integration"
        elif "-----  Hardware  ----------" in line:
            current_section = "hardware"
        elif "-----  Stress  ------------" in line:
            current_section = "stress"
        elif "|" in line and current_section and not line.startswith("+"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[1] and parts[2]:
                test_name = parts[1]
                result = parts[2]
                if current_section not in parsed["test_results"]:
                    parsed["test_results"][current_section] = {}
                parsed["test_results"][current_section][test_name] = result
    all_results = []
    for section in parsed["test_results"].values():
        all_results.extend(section.values())
    if all_results:
        if all("Pass" in result for result in all_results):
            parsed["overall_status"] = "Pass"
        elif any("Fail" in result for result in all_results):
            parsed["overall_status"] = "Fail"
        else:
            parsed["overall_status"] = "Mixed"
    return parsed


def extract_gpu_metrics():
    logger.info("Extracting GPU metrics...")
    metrics = {}
    success, stdout, stderr = run_command("which nvidia-smi")
    if not success:
        logger.info("No NVIDIA GPUs detected (nvidia-smi not available)")
        return metrics
    success, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
    )
    if success:
        lines = stdout.strip().split("\n")
        logger.info(f"Found {len(lines)} GPU(s)")
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                metrics[f"gpu_{i}"] = {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                }
                logger.info(f"GPU {i}: {parts[0]} - {parts[2]}MB total memory")
    else:
        logger.error(f"Failed to get GPU info: {stderr}")
    success, stdout, stderr = run_command(
        "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits"
    )
    if success:
        lines = stdout.strip().split("\n")
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and f"gpu_{i}" in metrics:
                metrics[f"gpu_{i}"].update(
                    {
                        "temperature_c": (
                            int(parts[0]) if parts[0] != "[Not Supported]" else None
                        ),
                        "utilization_percent": (
                            int(parts[1]) if parts[1] != "[Not Supported]" else None
                        ),
                        "power_draw_w": (
                            float(parts[2]) if parts[2] != "[Not Supported]" else None
                        ),
                    }
                )
    return metrics


def extract_system_metrics():
    logger.info("Extracting system metrics...")
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(
        f"System: {memory.available / (1024**3):.1f}GB RAM available, {cpu_percent:.1f}% CPU usage"
    )
    return {
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_percent": round((disk.used / disk.total) * 100, 2),
        },
        "cpu": {"usage_percent": cpu_percent, "count": psutil.cpu_count()},
    }


def extract_network_metrics(world_size, rank, current_host_rank, gpus_per_host):
    logger.info("Checking network fabric (EFA)...")
    metrics = {}

    # Check for AWS OFI NCCL plugin
    success, stdout, stderr = run_command(
        "find /opt/amazon -name 'libnccl-net.so' 2>/dev/null"
    )
    if success and stdout.strip():
        metrics["aws_ofi_nccl_plugin"] = {
            "installed": True,
            "path": stdout.strip().split("\n")[0],
        }
        logger.info(f"AWS OFI NCCL plugin found: {stdout.strip().split()[0]}")
    else:
        metrics["aws_ofi_nccl_plugin"] = {"installed": False}
        logger.warning("AWS OFI NCCL plugin not found - NCCL may not use EFA")

    success, stdout, stderr = run_command("fi_info")
    if success:
        metrics["efa_available"] = True
        efa_providers = []
        for line in stdout.split("\n"):
            if "provider: efa" in line:
                efa_providers.append(line.strip())
        metrics["efa_providers"] = efa_providers[:5]
        metrics["efa_provider_count"] = len(
            [l for l in stdout.split("\n") if "provider: efa" in l]
        )
        logger.info("EFA network fabric detected")
        if world_size > 1 and metrics["efa_provider_count"] > 0:
            first_rank_on_node = current_host_rank * gpus_per_host
            if MPI_AVAILABLE:
                comm.Barrier()
            if rank == first_rank_on_node:
                logger.info(
                    f"EFA testing: rank {rank} is first rank on node {current_host_rank}"
                )
                if current_host_rank == 0:
                    logger.info("Starting EFA server...")
                    success, stdout, stderr = run_command(
                        "timeout 60 /opt/amazon/efa/bin/fi_pingpong -p efa"
                    )
                    role = "server"
                else:
                    time.sleep(5)
                    master_hostname = os.environ.get("SM_MASTER_ADDR", "algo-1")
                    master_ip = get_ip_from_host(master_hostname)
                    logger.info(f"Connecting EFA client to {master_ip}...")
                    success, stdout, stderr = run_command(
                        f"timeout 45 /opt/amazon/efa/bin/fi_pingpong -p efa {master_ip}"
                    )
                    role = "client"
                metrics["efa_communication_test"] = {
                    "success": success,
                    "raw_output": stdout if success else stderr,
                    "test_role": role,
                    "rank": rank,
                }
                logger.info(
                    f"Rank {rank}: EFA {role} test completed - success: {success}"
                )
            else:
                logger.info(f"Rank {rank} skipping EFA test (not first rank on node)")
                metrics["efa_communication_test"] = {
                    "success": True,
                    "message": "EFA test handled by first rank on node",
                    "rank": rank,
                }
            if MPI_AVAILABLE:
                comm.Barrier()
                logger.info(f"Rank {rank}: EFA test phase completed")
        else:
            logger.info("EFA communication testing skipped")
            metrics["efa_communication_test"] = {
                "success": True,
                "message": "EFA availability confirmed - communication testing skipped",
            }
    else:
        metrics["efa_available"] = False
        metrics["efa_error"] = stderr
        logger.warning("EFA network fabric not available")
    return metrics


def extract_dcgm_metrics(
    world_size,
    rank,
    current_host_rank,
    gpus_per_host,
    enable_level1=True,
    enable_level3=True,
):
    logger.info("Running DCGM GPU diagnostics...")
    metrics = {"dcgm_available": False, "diagnostics": {}}
    if world_size > 1 and MPI_AVAILABLE:
        comm.Barrier()
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logger.info("No GPUs detected - skipping DCGM diagnostics")
        metrics["diagnostics"]["no_gpu"] = {
            "success": True,
            "message": "No GPUs available for DCGM diagnostics",
        }
        return metrics
    if world_size > 1:
        first_rank_on_node = current_host_rank * gpus_per_host
        if rank != first_rank_on_node:
            logger.info(
                f"Rank {rank} skipping DCGM (handled by rank {first_rank_on_node})"
            )
            metrics["diagnostics"]["delegated"] = {
                "success": True,
                "message": f"DCGM diagnostics handled by rank {first_rank_on_node}",
                "rank": rank,
            }
        else:
            success, stdout, stderr = run_command("which dcgmi")
            if success:
                metrics["dcgm_available"] = True
                logger.info("DCGM found, running diagnostics...")
                if enable_level1:
                    logger.info("Running DCGM Level 1 diagnostics...")
                    success, stdout, stderr = run_command("dcgmi diag -r 1")
                    metrics["diagnostics"]["basic"] = {
                        "success": success,
                        "parsed_output": parse_dcgm_output(stdout) if success else None,
                        "raw_output": stdout if success else stderr,
                    }
                    if success:
                        logger.info("DCGM Level 1 diagnostics completed successfully")
                    else:
                        logger.error(f"DCGM Level 1 diagnostics failed: {stderr}")
                else:
                    logger.info("DCGM Level 1 diagnostics skipped")
                    metrics["diagnostics"]["basic"] = {
                        "success": True,
                        "skipped": True,
                        "message": "Level 1 diagnostics disabled",
                    }
                if enable_level3:
                    logger.info("Running DCGM Level 3 diagnostics...")
                    success, stdout, stderr = run_command("dcgmi diag -r 3")
                    has_output = "Successfully ran diagnostic" in stdout
                    parsed = parse_dcgm_output(stdout) if has_output else None
                    metrics["diagnostics"]["extended"] = {
                        "success": success or has_output,
                        "parsed_output": parsed,
                        "raw_output": stdout if has_output else stderr,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code_zero": success,
                    }
                    if success:
                        logger.info("DCGM Level 3 diagnostics completed successfully")
                    elif has_output:
                        logger.warning(
                            "DCGM Level 3 diagnostics completed with warnings/failures. Check parsed_output for details."
                        )
                    else:
                        logger.warning(
                            f"DCGM Level 3 diagnostics failed or timed out. Exit code: non-zero. "
                            f"Stderr: {stderr[:500] if stderr else '(empty)'}. "
                            f"Stdout: {stdout[:500] if stdout else '(empty)'}"
                        )
                else:
                    logger.info("DCGM Level 3 diagnostics skipped")
                    metrics["diagnostics"]["extended"] = {
                        "success": True,
                        "skipped": True,
                        "message": "Level 3 diagnostics disabled",
                    }
            else:
                logger.warning("DCGM not available")
    else:
        success, stdout, stderr = run_command("which dcgmi")
        if success:
            metrics["dcgm_available"] = True
            logger.info("DCGM found, running diagnostics...")
            if enable_level1:
                logger.info("Running DCGM Level 1 diagnostics...")
                success, stdout, stderr = run_command("dcgmi diag -r 1")
                metrics["diagnostics"]["basic"] = {
                    "success": success,
                    "parsed_output": parse_dcgm_output(stdout) if success else None,
                    "raw_output": stdout if success else stderr,
                }
                if success:
                    logger.info("DCGM Level 1 diagnostics completed successfully")
                else:
                    logger.error(f"DCGM Level 1 diagnostics failed: {stderr}")
            else:
                logger.info("DCGM Level 1 diagnostics skipped")
                metrics["diagnostics"]["basic"] = {
                    "success": True,
                    "skipped": True,
                    "message": "Level 1 diagnostics disabled",
                }
            if enable_level3:
                logger.info("Running DCGM Level 3 diagnostics...")
                success, stdout, stderr = run_command("dcgmi diag -r 3")
                has_output = "Successfully ran diagnostic" in stdout
                parsed = parse_dcgm_output(stdout) if has_output else None
                metrics["diagnostics"]["extended"] = {
                    "success": success or has_output,
                    "parsed_output": parsed,
                    "raw_output": stdout if has_output else stderr,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code_zero": success,
                }
                if success:
                    logger.info("DCGM Level 3 diagnostics completed successfully")
                elif has_output:
                    logger.warning(
                        "DCGM Level 3 diagnostics completed with warnings/failures. Check parsed_output for details."
                    )
                else:
                    logger.warning(
                        f"DCGM Level 3 diagnostics failed or timed out. Exit code: non-zero. "
                        f"Stderr: {stderr[:500] if stderr else '(empty)'}. "
                        f"Stdout: {stdout[:500] if stdout else '(empty)'}"
                    )
            else:
                logger.info("DCGM Level 3 diagnostics skipped")
                metrics["diagnostics"]["extended"] = {
                    "success": True,
                    "skipped": True,
                    "message": "Level 3 diagnostics disabled",
                }
        else:
            logger.warning("DCGM not available")
    if world_size > 1 and MPI_AVAILABLE:
        comm.Barrier()
        logger.info(f"Rank {rank}: DCGM test phase completed")
    return metrics


def generate_summary_and_recommendations(
    gpu_metrics, network_metrics, dcgm_metrics, world_size
):
    summary = {
        "overall_status": "PASS",
        "test_results": {},
        "issues": [],
        "recommendations": [],
    }
    summary["test_results"]["gpu_detection"] = len(gpu_metrics) > 0
    summary["test_results"]["efa_network"] = network_metrics.get("efa_available", False)
    if dcgm_metrics.get("aggregated"):
        dcgm_basic_success = False
        dcgm_extended_success = False
        for node_metrics in dcgm_metrics.get("nodes", {}).values():
            if node_metrics.get("diagnostics", {}).get("basic", {}).get("success"):
                dcgm_basic_success = True
            if node_metrics.get("diagnostics", {}).get("extended", {}).get("success"):
                dcgm_extended_success = True
        summary["test_results"]["dcgm_basic"] = dcgm_basic_success
        summary["test_results"]["dcgm_extended"] = dcgm_extended_success
    else:
        summary["test_results"]["dcgm_basic"] = (
            dcgm_metrics.get("diagnostics", {}).get("basic", {}).get("success", False)
        )
        summary["test_results"]["dcgm_extended"] = (
            dcgm_metrics.get("diagnostics", {})
            .get("extended", {})
            .get("success", False)
        )
    if len(gpu_metrics) == 0:
        logger.info("No GPUs detected - CPU-only instance")
        summary["recommendations"].append(
            "CPU-only instance - no GPU validation needed"
        )
    elif not summary["test_results"]["gpu_detection"]:
        summary["overall_status"] = "FAIL"
        summary["issues"].append("GPU detection failed")
        summary["recommendations"].append("Check GPU drivers and CUDA installation")
    if (
        world_size > 1
        and len(gpu_metrics) > 0
        and not summary["test_results"]["efa_network"]
    ):
        summary["issues"].append("EFA network not available for multi-node training")
        summary["recommendations"].append(
            "Use EFA-enabled instance types (e.g., p4d, p5) for multi-node training"
        )
    if len(gpu_metrics) > 0:
        dcgm_delegated = (
            dcgm_metrics.get("diagnostics", {})
            .get("delegated", {})
            .get("success", False)
        )
        if not dcgm_delegated and not dcgm_metrics.get("aggregated"):
            if not summary["test_results"]["dcgm_basic"]:
                summary["issues"].append("DCGM basic diagnostics failed")
                summary["recommendations"].append(
                    "Check GPU health and DCGM installation"
                )
            if not summary["test_results"]["dcgm_extended"]:
                summary["issues"].append(
                    "DCGM extended diagnostics failed or timed out"
                )
                summary["recommendations"].append(
                    "Consider running extended diagnostics separately for detailed GPU validation"
                )
    if summary["issues"] and summary["overall_status"] != "FAIL":
        summary["overall_status"] = "WARN"
    if not summary["issues"]:
        summary["recommendations"].append("Cluster is ready for distributed training")
    return summary


def save_metrics_to_shared_file(
    all_metrics,
    summary,
    rank,
    world_size,
    current_host_rank,
    host_count,
    gpus_per_host,
    metrics_dir="/opt/ml/output/data/metrics",
):
    logger.info(f"Rank {rank}: Entering save_metrics_to_shared_file")
    # Calculate actual processes per node based on world_size
    processes_per_node = world_size // host_count
    first_rank_on_node = current_host_rank * processes_per_node
    if rank == first_rank_on_node:
        node_metrics = {
            "timestamp": all_metrics["timestamp"],
            "node_info": all_metrics["node_info"],
            "summary": summary,
            "gpu": all_metrics["gpu"],
            "system": all_metrics["system"],
            "network": all_metrics["network"],
            "dcgm": all_metrics["dcgm"],
        }
        if rank != 0 and MPI_AVAILABLE:
            logger.info(f"Rank {rank}: Sending metrics to rank 0")
            comm.send(node_metrics, dest=0, tag=rank)
            logger.info(f"Rank {rank}: Metrics sent successfully")
        else:
            logger.info(f"Rank 0: Starting to collect from {host_count-1} nodes")
            all_node_metrics = {f"node_{current_host_rank}": node_metrics}
            if MPI_AVAILABLE:
                for node_rank in range(1, host_count):
                    first_rank = node_rank * processes_per_node
                    logger.info(f"Rank 0: Waiting to receive from rank {first_rank}")
                    node_data = comm.recv(source=first_rank, tag=first_rank)
                    all_node_metrics[f"node_{node_rank}"] = node_data
                    logger.info(
                        f"Rank 0: Successfully received metrics from rank {first_rank}"
                    )
            cluster_context = {
                "data_completeness": "complete",
                "note": f"Data from all {host_count} nodes",
                "expected_total_gpus": world_size,
                "cluster_topology": {
                    "total_nodes": host_count,
                    "gpus_per_node": gpus_per_host,
                    "processes_per_node": gpus_per_host,
                },
            }
            final_metrics = {
                "timestamp": all_metrics["timestamp"],
                "cluster_info": cluster_context,
                "cluster_summary": summary,
                "nodes": all_node_metrics,
            }
            os.makedirs(metrics_dir, exist_ok=True)
            final_path = os.path.join(metrics_dir, "health_check_metrics.json")
            with open(final_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
            logger.info(f"Metrics saved to {final_path}")
    if world_size > 1 and MPI_AVAILABLE:
        logger.info(f"Rank {rank}: Entering final barrier for metrics saving")
        sys.stdout.flush()
        comm.Barrier()
        logger.info(f"Rank {rank}: All ranks synchronized after metrics saving")
        sys.stdout.flush()


def log_health_check_summary(
    node_info,
    summary,
    gpu_metrics,
    system_metrics,
    network_metrics,
    dcgm_metrics,
    rank,
    world_size,
):
    logger.info("=== Pre-flight Check Summary ===")
    logger.info(
        f"Node: {node_info['current_host']} (host rank {node_info['host_rank']}/{node_info['host_count']}, process rank {rank}/{world_size})"
    )
    logger.info(f"Overall Status: {summary['overall_status']}")
    logger.info(f"GPUs detected: {len(gpu_metrics)}")
    if gpu_metrics:
        logger.info("=== GPU Details ===")
        for gpu_id, gpu_data in gpu_metrics.items():
            logger.info(
                f"{gpu_id}: {gpu_data['name']} - {gpu_data['memory_total_mb']}MB total, {gpu_data['temperature_c']}°C, {gpu_data['utilization_percent']}% util, {gpu_data['power_draw_w']}W"
            )
    logger.info(
        f"System memory: {system_metrics['memory']['available_gb']:.1f}GB available"
    )
    logger.info(f"CPU usage: {system_metrics['cpu']['usage_percent']:.1f}%")
    logger.info(f"EFA available: {network_metrics['efa_available']}")
    logger.info(
        f"DCGM available: {dcgm_metrics.get('dcgm_available', dcgm_metrics.get('aggregated', False))}"
    )
    if summary["issues"]:
        logger.warning("Issues found:")
        for issue in summary["issues"]:
            logger.warning(f"  - {issue}")
    if summary["recommendations"]:
        logger.info("Recommendations:")
        for rec in summary["recommendations"]:
            logger.info(f"  - {rec}")


def setup_mlflow(mlflow_uri, mlflow_experiment_name):
    """Set up MLflow tracking.

    Args:
        mlflow_uri: MLflow tracking URI
        mlflow_experiment_name: MLflow experiment name

    Returns:
        bool: True if setup successful, False otherwise
    """
    logger.info("Initializing MLflow")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    current_datetime = datetime.now(timezone.utc)
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    run_name = f"health-check-{formatted_datetime}"

    mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run started: {run_name}")
    return True


def export_to_mlflow(cluster_metrics, cluster_summary):
    """Export health check metrics to MLflow.

    Args:
        cluster_metrics: Aggregated cluster metrics dictionary (from JSON file)
        cluster_summary: Cluster-wide health check summary
    """
    logger.info("Exporting metrics to MLflow...")

    # Log cluster-level parameters
    mlflow.log_params(
        {
            "total_nodes": cluster_metrics.get("cluster_info", {})
            .get("cluster_topology", {})
            .get("total_nodes", 0),
            "gpus_per_node": cluster_metrics.get("cluster_info", {})
            .get("cluster_topology", {})
            .get("gpus_per_node", 0),
            "expected_total_gpus": cluster_metrics.get("cluster_info", {}).get(
                "expected_total_gpus", 0
            ),
            "overall_status": cluster_summary["overall_status"],
        }
    )

    # Log test results
    for test_name, result in cluster_summary["test_results"].items():
        mlflow.log_metric(f"test_{test_name}", int(result))

    # Log performance metrics
    if cluster_summary.get("performance_summary"):
        for metric_name, value in cluster_summary["performance_summary"].items():
            mlflow.log_metric(metric_name, value)

    # Log metrics from all nodes
    for node_name, node_data in cluster_metrics.get("nodes", {}).items():
        node_prefix = node_name

        # Log GPU metrics per node
        for gpu_id, gpu_data in node_data.get("gpu", {}).items():
            mlflow.log_metrics(
                {
                    f"{node_prefix}_{gpu_id}_temperature_c": gpu_data.get(
                        "temperature_c", 0
                    )
                    or 0,
                    f"{node_prefix}_{gpu_id}_utilization_percent": gpu_data.get(
                        "utilization_percent", 0
                    )
                    or 0,
                    f"{node_prefix}_{gpu_id}_memory_used_mb": gpu_data[
                        "memory_used_mb"
                    ],
                }
            )

        # Log system metrics per node
        mlflow.log_metrics(
            {
                f"{node_prefix}_system_memory_available_gb": node_data["system"][
                    "memory"
                ]["available_gb"],
                f"{node_prefix}_system_cpu_usage_percent": node_data["system"]["cpu"][
                    "usage_percent"
                ],
            }
        )

    # Log full metrics as artifact
    mlflow.log_dict(cluster_metrics, "health_check_metrics.json")

    logger.info("Metrics exported to MLflow successfully")


def run_health_checks(
    metrics_dir="/opt/ml/output/data/metrics",
    enable_gpu_check=True,
    enable_system_check=True,
    enable_network_check=True,
    enable_dcgm_level1=True,
    enable_dcgm_level3=True,
    export_mlflow=False,
    mlflow_uri=None,
    mlflow_experiment_name="sagemaker-health-checks",
):
    logger.info("Starting comprehensive pre-flight health checks")
    host_count = int(os.environ.get("SM_HOST_COUNT", "1"))
    current_host_rank = int(os.environ.get("SM_CURRENT_HOST_RANK", "0"))
    gpus_per_host = int(os.environ.get("SM_NUM_GPUS", "1"))
    gpu_metrics = extract_gpu_metrics() if enable_gpu_check else {}
    system_metrics = extract_system_metrics() if enable_system_check else {}
    network_metrics = (
        extract_network_metrics(world_size, rank, current_host_rank, gpus_per_host)
        if enable_network_check
        else {}
    )
    dcgm_metrics = (
        extract_dcgm_metrics(
            world_size,
            rank,
            current_host_rank,
            gpus_per_host,
            enable_dcgm_level1,
            enable_dcgm_level3,
        )
        if (enable_dcgm_level1 or enable_dcgm_level3)
        else {}
    )
    node_info = {
        "world_size": world_size,
        "rank": rank,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "current_host": os.environ.get("SM_CURRENT_HOST", "unknown"),
        "host_rank": current_host_rank,
        "host_count": host_count,
        "gpus_per_host": gpus_per_host,
    }
    summary = generate_summary_and_recommendations(
        gpu_metrics, network_metrics, dcgm_metrics, world_size
    )
    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_info": node_info,
        "summary": summary,
        "gpu": gpu_metrics,
        "system": system_metrics,
        "network": network_metrics,
        "dcgm": dcgm_metrics,
    }
    save_metrics_to_shared_file(
        all_metrics,
        summary,
        rank,
        world_size,
        current_host_rank,
        host_count,
        gpus_per_host,
        metrics_dir,
    )
    log_health_check_summary(
        node_info,
        summary,
        gpu_metrics,
        system_metrics,
        network_metrics,
        dcgm_metrics,
        rank,
        world_size,
    )

    # Export to MLflow if enabled (only rank 0)
    if export_mlflow and rank == 0:
        # Wait for metrics file to be written
        time.sleep(1)
        metrics_file = os.path.join(metrics_dir, "health_check_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                cluster_metrics = json.load(f)
            cluster_summary = cluster_metrics.get("cluster_summary", summary)
            if setup_mlflow(mlflow_uri, mlflow_experiment_name):
                export_to_mlflow(cluster_metrics, cluster_summary)
                mlflow.end_run()
        else:
            logger.warning(
                f"Metrics file not found: {metrics_file}, skipping MLflow export"
            )

    logger.info(f"Rank {rank}: run_health_checks completed, returning metrics")
    sys.stdout.flush()
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker Training Cluster Health Checks (MPI)"
    )
    parser.add_argument(
        "--metrics-dir", type=str, default="/opt/ml/output/data/metrics"
    )
    parser.add_argument("--gpu-check", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument(
        "--system-check", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--network-check", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--dcgm-level1", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--dcgm-level3", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--export-mlflow",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Export metrics to MLflow (default: false)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: None)",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="sagemaker-health-checks",
        help="MLflow experiment name (default: sagemaker-health-checks)",
    )
    args = parser.parse_args()
    logger.info("Starting SageMaker Training Cluster Pre-flight Health Checks (MPI)")
    run_health_checks(
        metrics_dir=args.metrics_dir,
        enable_gpu_check=args.gpu_check,
        enable_system_check=args.system_check,
        enable_network_check=args.network_check,
        enable_dcgm_level1=args.dcgm_level1,
        enable_dcgm_level3=args.dcgm_level3,
        export_mlflow=args.export_mlflow,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
    logger.info("Pre-flight health checks completed successfully")


if __name__ == "__main__":
    main()
