#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone
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
import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_gpu_count():
    """Get the number of NVIDIA GPUs available on the system.

    Returns:
        int: Number of GPUs detected, 0 if none or error
    """
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
    """Resolve hostname to IP address with retry logic.

    Args:
        host: Hostname to resolve

    Returns:
        str: IP address

    Raises:
        Exception: If hostname cannot be resolved within timeout
    """
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
    """Execute shell command and capture output.

    Args:
        cmd: Shell command to execute

    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def parse_nccl_output(output):
    """Parse NCCL test output to extract performance metrics.

    Args:
        output: Raw NCCL test output string

    Returns:
        dict: Parsed metrics including version, devices, bandwidth, and test summary
    """
    parsed = {
        "nccl_version": None,
        "devices": [],
        "avg_bandwidth_gbps": None,
        "peak_bandwidth_gbps": None,
        "test_summary": [],
    }

    lines = output.split("\n")
    for line in lines:
        # Extract NCCL version
        if "NCCL version" in line:
            parsed["nccl_version"] = line.strip()

        # Extract device info
        if "Rank" in line and "device" in line and "NVIDIA" in line:
            parsed["devices"].append(line.strip())

        # Extract average bandwidth using regex
        avg_match = re.search(r"Avg bus bandwidth\s*:\s*(\d+(?:\.\d+)?)", line)
        if avg_match:
            parsed["avg_bandwidth_gbps"] = float(avg_match.group(1))

        # Extract performance data using regex - match actual NCCL format
        perf_match = re.match(
            r"\s*(\d+)\s+\d+\s+\w+\s+\w+\s+[-\d]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+",
            line,
        )
        if perf_match:
            try:
                size = perf_match.group(1)
                algbw = float(perf_match.group(2))  # out-of-place algbw
                busbw_out = float(perf_match.group(3))  # out-of-place busbw
                busbw_in = float(perf_match.group(6))  # in-place busbw (usually higher)

                # Use the higher bandwidth value
                busbw = max(busbw_out, busbw_in)

                parsed["test_summary"].append(
                    {"size": size, "algbw_gbps": algbw, "busbw_gbps": busbw}
                )
                if (
                    parsed["peak_bandwidth_gbps"] is None
                    or busbw > parsed["peak_bandwidth_gbps"]
                ):
                    parsed["peak_bandwidth_gbps"] = busbw
            except (ValueError, IndexError):
                pass

    # Keep only last 5 test results for readability
    parsed["test_summary"] = parsed["test_summary"][-5:]

    return parsed


def parse_dcgm_output(output):
    """Parse DCGM diagnostic output to extract test results.

    Args:
        output: Raw DCGM diagnostic output string

    Returns:
        dict: Parsed results including version, test results, and overall status
    """
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

        # Extract metadata
        if "DCGM Version" in line:
            parsed["dcgm_version"] = line.split("|")[2].strip()
        elif "Driver Version Detected" in line:
            parsed["driver_version"] = line.split("|")[2].strip()
        elif "GPU Device IDs Detected" in line:
            parsed["gpu_device_ids"] = line.split("|")[2].strip()

        # Track sections
        elif "-----  Deployment  --------" in line:
            current_section = "deployment"
        elif "-----  Integration  -------" in line:
            current_section = "integration"
        elif "-----  Hardware  ----------" in line:
            current_section = "hardware"
        elif "-----  Stress  ------------" in line:
            current_section = "stress"

        # Extract test results
        elif "|" in line and current_section and not line.startswith("+"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[1] and parts[2]:
                test_name = parts[1]
                result = parts[2]
                if current_section not in parsed["test_results"]:
                    parsed["test_results"][current_section] = {}
                parsed["test_results"][current_section][test_name] = result

    # Determine overall status
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


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def extract_gpu_metrics():
    """Extract GPU metrics using nvidia-smi.

    Returns:
        dict: GPU metrics per device including name, memory, temperature, utilization, and power
    """
    logger.info("Extracting GPU metrics...")
    metrics = {}

    # Check if nvidia-smi is available (some instances don't have GPUs)
    success, stdout, stderr = run_command("which nvidia-smi")
    if not success:
        logger.info("No NVIDIA GPUs detected (nvidia-smi not available)")
        return metrics

    # GPU basic info
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

    # GPU temperature and utilization
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
    """Extract system resource metrics using psutil.

    Returns:
        dict: System metrics including memory, disk, and CPU usage
    """
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
    """Extract network fabric information and test EFA connectivity.

    Args:
        world_size: Total number of processes
        rank: Current process rank
        current_host_rank: Current host/node rank
        gpus_per_host: Number of GPUs per host

    Returns:
        dict: Network metrics including EFA availability and communication test results
    """
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

    # EFA providers
    success, stdout, stderr = run_command("fi_info")
    if success:
        metrics["efa_available"] = True
        # Extract key EFA info instead of full output
        efa_providers = []
        for line in stdout.split("\n"):
            if "provider: efa" in line:
                efa_providers.append(line.strip())
        metrics["efa_providers"] = efa_providers[:5]  # Limit to first 5
        metrics["efa_provider_count"] = len(
            [l for l in stdout.split("\n") if "provider: efa" in l]
        )
        logger.info("EFA network fabric detected")

        # Test EFA communication with one process per node
        if world_size > 1 and metrics["efa_provider_count"] > 0:
            # Only test EFA between first rank of each node
            first_rank_on_node = current_host_rank * gpus_per_host

            # All ranks synchronize before EFA test
            if dist.is_initialized():
                dist.barrier()

            if rank == first_rank_on_node:
                logger.info(
                    f"EFA testing: rank {rank} is first rank on node {current_host_rank}"
                )

                if current_host_rank == 0:
                    # First node: run server
                    logger.info("Starting EFA server...")
                    success, stdout, stderr = run_command(
                        "timeout 60 /opt/amazon/efa/bin/fi_pingpong -p efa"
                    )
                    role = "server"
                else:
                    # Other nodes: run client after brief delay
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
                # Other ranks on same node skip EFA testing
                logger.info(f"Rank {rank} skipping EFA test (not first rank on node)")
                metrics["efa_communication_test"] = {
                    "success": True,
                    "message": "EFA test handled by first rank on node",
                    "rank": rank,
                }

            # All ranks synchronize after EFA test
            if dist.is_initialized():
                dist.barrier()
                logger.info(
                    f"Rank {rank}: EFA test phase completed, proceeding to NCCL tests"
                )
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


def extract_nccl_metrics(world_size, rank):
    """Test multi-GPU communication using NCCL via torch.distributed.

    Args:
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        dict: NCCL test results including bandwidth and latency measurements
    """
    logger.info("Running NCCL communication tests...")
    metrics = {"nccl_tests_available": True, "communication_tests": {}}

    # Get GPU count
    gpu_count = get_gpu_count()

    logger.info(f"Detected {gpu_count} GPUs, world size: {world_size}, rank: {rank}")

    if gpu_count == 0:
        logger.info("No GPUs detected - skipping NCCL tests")
        metrics["communication_tests"]["no_gpu"] = {
            "success": True,
            "test_type": "no-gpu",
            "message": "No GPUs available for NCCL testing",
        }
        return metrics

    # Test NCCL via torch.distributed (already initialized by torchrun)
    if world_size > 1:
        logger.info(
            f"Rank {rank}: Testing NCCL communication across {world_size} processes"
        )

        try:
            # Set device for this rank
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(f"cuda:{local_rank}")

            # Test all_reduce with different sizes
            test_sizes = [1024, 1024 * 1024, 128 * 1024 * 1024]  # 1KB, 1MB, 128MB
            results = []

            for size_bytes in test_sizes:
                size_elements = size_bytes // 4  # float32
                tensor = torch.ones(size_elements, device=device)

                # Warmup
                for _ in range(3):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()

                # Timed test
                start = time.time()
                iterations = 10
                for _ in range(iterations):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                elapsed = time.time() - start

                bandwidth_gbps = (size_bytes * iterations) / elapsed / 1e9
                results.append(
                    {
                        "size_bytes": size_bytes,
                        "bandwidth_gbps": round(bandwidth_gbps, 2),
                        "latency_ms": round(elapsed * 1000 / iterations, 2),
                    }
                )

            metrics["communication_tests"]["distributed_all_reduce"] = {
                "success": True,
                "test_type": "distributed",
                "processes_tested": world_size,
                "results": results,
                "peak_bandwidth_gbps": max(r["bandwidth_gbps"] for r in results),
            }
            logger.info(f"Rank {rank}: NCCL all_reduce test completed")

        except Exception as e:
            logger.error(f"Rank {rank}: NCCL test failed: {e}")
            metrics["communication_tests"]["distributed_all_reduce"] = {
                "success": False,
                "error": str(e),
                "test_type": "distributed",
            }

        # Synchronize all ranks after NCCL tests
        if dist.is_initialized():
            dist.barrier()
            logger.info(f"Rank {rank}: NCCL test phase completed")
    else:
        # Single process: test local multi-GPU if available
        if gpu_count > 1:
            logger.info(
                f"Single process: testing multi-GPU communication with {gpu_count} GPUs"
            )
            try:
                # Initialize process group for single-node multi-GPU
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl",
                        init_method="tcp://localhost:29500",
                        world_size=1,
                        rank=0,
                    )

                device = torch.device("cuda:0")
                size_bytes = 128 * 1024 * 1024
                size_elements = size_bytes // 4
                tensor = torch.ones(size_elements, device=device)

                start = time.time()
                iterations = 10
                for _ in range(iterations):
                    # Copy across GPUs
                    for i in range(1, gpu_count):
                        tensor_copy = tensor.to(f"cuda:{i}")
                torch.cuda.synchronize()
                elapsed = time.time() - start

                bandwidth_gbps = (
                    (size_bytes * iterations * (gpu_count - 1)) / elapsed / 1e9
                )

                metrics["communication_tests"]["multi_gpu"] = {
                    "success": True,
                    "test_type": "multi-gpu",
                    "gpus_tested": gpu_count,
                    "bandwidth_gbps": round(bandwidth_gbps, 2),
                }
                logger.info("Multi-GPU communication test completed")
            except Exception as e:
                logger.error(f"Multi-GPU test failed: {e}")
                metrics["communication_tests"]["multi_gpu"] = {
                    "success": False,
                    "error": str(e),
                }
        else:
            logger.info("Single GPU: no NCCL communication to test")
            metrics["communication_tests"]["single_gpu"] = {
                "success": True,
                "test_type": "single-gpu",
                "message": "No multi-GPU communication available with 1 GPU",
            }

    return metrics


def extract_dcgm_metrics(
    world_size,
    rank,
    current_host_rank,
    gpus_per_host,
    enable_level1=True,
    enable_level3=True,
):
    """Run DCGM GPU diagnostics (basic and extended).

    Args:
        world_size: Total number of processes
        rank: Current process rank
        current_host_rank: Current host/node rank
        gpus_per_host: Number of GPUs per host

    Returns:
        dict: DCGM diagnostic results including test status and parsed output
    """
    logger.info("Running DCGM GPU diagnostics...")
    metrics = {"dcgm_available": False, "diagnostics": {}}

    # Synchronize all ranks before DCGM tests
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    # Check if we have GPUs first
    gpu_count = get_gpu_count()

    if gpu_count == 0:
        logger.info("No GPUs detected - skipping DCGM diagnostics")
        metrics["diagnostics"]["no_gpu"] = {
            "success": True,
            "message": "No GPUs available for DCGM diagnostics",
        }
        return metrics

    # Only run DCGM on first rank per node to avoid conflicts
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
            # Don't return early - need to reach the barrier at the end
        else:
            # This rank runs DCGM
            # Check if DCGM is available
            success, stdout, stderr = run_command("which dcgmi")
            if success:
                metrics["dcgm_available"] = True
                logger.info("DCGM found, running diagnostics...")

                # Run Level 1 diagnostics if enabled
                if enable_level1:
                    logger.info(
                        "Running DCGM Level 1 diagnostics (no timeout - may take several minutes)..."
                    )
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

                # Run Level 3 diagnostics if enabled
                if enable_level3:
                    logger.info(
                        "Running DCGM Level 3 diagnostics (no timeout - may take several minutes)..."
                    )
                    success, stdout, stderr = run_command("dcgmi diag -r 3")
                    # DCGM returns non-zero exit code if any test fails, but still provides output
                    # Check if we got valid output even if exit code is non-zero
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
                            f"DCGM Level 3 diagnostics completed with warnings/failures. "
                            f"Check parsed_output for details."
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
        # Single process - run DCGM directly
        success, stdout, stderr = run_command("which dcgmi")
        if success:
            metrics["dcgm_available"] = True
            logger.info("DCGM found, running diagnostics...")

            # Run Level 1 diagnostics if enabled
            if enable_level1:
                logger.info(
                    "Running DCGM Level 1 diagnostics (no timeout - may take several minutes)..."
                )
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

            # Run Level 3 diagnostics if enabled
            if enable_level3:
                logger.info(
                    "Running DCGM Level 3 diagnostics (no timeout - may take several minutes)..."
                )
                success, stdout, stderr = run_command("dcgmi diag -r 3")
                # DCGM returns non-zero exit code if any test fails, but still provides output
                # Check if we got valid output even if exit code is non-zero
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
                        f"DCGM Level 3 diagnostics completed with warnings/failures. "
                        f"Check parsed_output for details."
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

    # Synchronize all ranks after DCGM tests
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        logger.info(f"Rank {rank}: DCGM test phase completed")

    return metrics


def extract_efa_performance(network_metrics):
    """Extract key EFA performance metrics from network test results.

    Args:
        network_metrics: Network metrics dictionary from extract_network_metrics

    Returns:
        dict: EFA performance metrics including bandwidth and message size
    """
    perf = {"available": network_metrics.get("efa_available", False)}

    if network_metrics.get("efa_communication_test", {}).get("success"):
        efa_output = network_metrics["efa_communication_test"].get("raw_output", "")
        if "MB/sec" in efa_output:
            lines = efa_output.split("\n")
            for line in lines:
                if "MB/sec" in line and not line.startswith("#"):
                    parts = line.split()
                    # Find the numeric value before 'MB/sec'
                    for i, part in enumerate(parts):
                        if part == "MB/sec" and i > 0:
                            try:
                                perf["bandwidth_mbps"] = float(parts[i - 1])
                                perf["message_size"] = parts[0]
                                return perf
                            except (ValueError, IndexError):
                                continue
    return perf


def extract_nccl_performance(nccl_metrics):
    """Extract key NCCL performance metrics from test results.

    Args:
        nccl_metrics: NCCL metrics dictionary from extract_nccl_metrics

    Returns:
        dict: NCCL performance metrics including peak bandwidth and test results
    """
    perf = {"available": nccl_metrics.get("nccl_tests_available", False)}

    if nccl_metrics.get("communication_tests"):
        for test_name, test_data in nccl_metrics["communication_tests"].items():
            if test_data.get("success"):
                # Handle new torch-based results format
                if test_data.get("results"):
                    perf[test_name] = {
                        "peak_bandwidth_gbps": test_data.get("peak_bandwidth_gbps"),
                        "results": test_data["results"],
                    }
                # Handle old parsed_output format (if still present)
                elif test_data.get("parsed_output"):
                    parsed = test_data["parsed_output"]
                    perf[test_name] = {
                        "peak_bandwidth_gbps": parsed.get("peak_bandwidth_gbps"),
                        "avg_bandwidth_gbps": parsed.get("avg_bandwidth_gbps"),
                    }
                    if parsed.get("test_summary") and len(parsed["test_summary"]) > 0:
                        largest = parsed["test_summary"][-1]
                        perf[test_name]["largest_message"] = {
                            "size": largest["size"],
                            "bandwidth_gbps": largest["busbw_gbps"],
                        }
    return perf


# ============================================================================
# BUSINESS LOGIC
# ============================================================================


def generate_summary_and_recommendations(
    gpu_metrics, network_metrics, nccl_metrics, dcgm_metrics, world_size
):
    """Generate health check summary and recommendations based on test results.

    Args:
        gpu_metrics: GPU metrics from extract_gpu_metrics
        network_metrics: Network metrics from extract_network_metrics
        nccl_metrics: NCCL metrics from extract_nccl_metrics
        dcgm_metrics: DCGM metrics from extract_dcgm_metrics
        world_size: Total number of processes

    Returns:
        dict: Summary including overall status, test results, issues, and recommendations
    """
    summary = {
        "overall_status": "PASS",
        "test_results": {},
        "cluster_info": {},
        "performance_summary": {},
        "issues": [],
        "recommendations": [],
    }

    # Test results
    summary["test_results"]["gpu_detection"] = len(gpu_metrics) > 0
    summary["test_results"]["efa_network"] = network_metrics.get("efa_available", False)
    summary["test_results"]["nccl_communication"] = nccl_metrics.get(
        "nccl_tests_available", False
    )

    # Handle aggregated DCGM metrics
    if dcgm_metrics.get("aggregated"):
        # Check if any node has DCGM results
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
        # Single node or non-aggregated
        summary["test_results"]["dcgm_basic"] = (
            dcgm_metrics.get("diagnostics", {}).get("basic", {}).get("success", False)
        )
        summary["test_results"]["dcgm_extended"] = (
            dcgm_metrics.get("diagnostics", {})
            .get("extended", {})
            .get("success", False)
        )

    # Cluster info
    if "cluster_info" in nccl_metrics:
        summary["cluster_info"] = nccl_metrics["cluster_info"]

    # Performance summary - extract key NCCL metrics
    if nccl_metrics.get("communication_tests"):
        perf = {}
        for test_name, test_data in nccl_metrics["communication_tests"].items():
            if test_data.get("success") and test_data.get("peak_bandwidth_gbps"):
                perf[f"{test_name}_bandwidth_gbps"] = test_data["peak_bandwidth_gbps"]
        summary["performance_summary"] = perf

    # Check for issues and generate recommendations
    if len(gpu_metrics) == 0:
        logger.info("No GPUs detected - CPU-only instance")
        summary["recommendations"].append(
            "CPU-only instance - no GPU validation needed"
        )
    elif not summary["test_results"]["gpu_detection"]:
        summary["overall_status"] = "FAIL"
        summary["issues"].append("GPU detection failed")
        summary["recommendations"].append("Check GPU drivers and CUDA installation")

    # Only check EFA for multi-node setups
    if (
        world_size > 1
        and len(gpu_metrics) > 0
        and not summary["test_results"]["efa_network"]
    ):
        summary["issues"].append("EFA network not available for multi-node training")
        summary["recommendations"].append(
            "Use EFA-enabled instance types (e.g., p4d, p5) for multi-node training"
        )

    # Only check NCCL if we have GPUs
    if len(gpu_metrics) > 0 and not summary["test_results"]["nccl_communication"]:
        summary["overall_status"] = "FAIL"
        summary["issues"].append("NCCL communication tests failed")
        summary["recommendations"].append(
            "Check NCCL installation and GPU accessibility"
        )

    # Only check DCGM if we have GPUs
    if len(gpu_metrics) > 0:
        # Check if DCGM was delegated to another rank
        dcgm_delegated = (
            dcgm_metrics.get("diagnostics", {})
            .get("delegated", {})
            .get("success", False)
        )

        if dcgm_delegated or dcgm_metrics.get("aggregated"):
            # DCGM was delegated or aggregated - check aggregated results
            pass
        else:
            # This rank should have run DCGM - check results
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

    # Performance recommendations
    if summary["performance_summary"]:
        for test_name, bandwidth in summary["performance_summary"].items():
            if bandwidth < 0.5:
                summary["issues"].append(f"Low {test_name}: {bandwidth:.2f} GB/s")
                summary["recommendations"].append(
                    f"Investigate {test_name} performance - check network and GPU configuration"
                )

    # Overall status
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
    local_rank,
    metrics_dir="/opt/ml/output/data/metrics",
):
    """Save metrics using distributed communication to aggregate to rank 0.

    Args:
        all_metrics: Complete metrics dictionary
        summary: Health check summary
        rank: Current process rank
        world_size: Total number of processes
        current_host_rank: Current host/node rank
        host_count: Total number of hosts/nodes
        gpus_per_host: Number of GPUs per host
        local_rank: Local rank within node
        metrics_dir: Directory to save metrics file
    """
    logger.info(f"Rank {rank}: Entering save_metrics_to_shared_file")

    # Each node's first rank prepares and sends metrics
    first_rank_on_node = current_host_rank * gpus_per_host

    if rank == first_rank_on_node:
        node_metrics = {
            "timestamp": all_metrics["timestamp"],
            "node_info": all_metrics["node_info"],
            "summary": summary,
            "gpu": all_metrics["gpu"],
            "system": all_metrics["system"],
            "network": all_metrics["network"],
            "nccl": all_metrics["nccl"],
            "dcgm": all_metrics["dcgm"],
        }

        if rank != 0:
            # Non-zero first ranks send to rank 0
            logger.info(f"Rank {rank}: Preparing to send metrics to rank 0")
            metrics_bytes = pickle.dumps(node_metrics)
            size_tensor = torch.tensor([len(metrics_bytes)], dtype=torch.long)
            if torch.cuda.is_available():
                size_tensor = size_tensor.to(f"cuda:{local_rank}")
            logger.info(
                f"Rank {rank}: Sending size tensor ({size_tensor.item()} bytes)"
            )
            dist.send(size_tensor, dst=0)
            logger.info(f"Rank {rank}: Size tensor sent, now sending data")

            data_tensor = torch.tensor(list(metrics_bytes), dtype=torch.uint8)
            if torch.cuda.is_available():
                data_tensor = data_tensor.to(f"cuda:{local_rank}")
            dist.send(data_tensor, dst=0)
            logger.info(f"Rank {rank}: Data tensor sent successfully")
        else:
            # Rank 0 collects from all nodes
            logger.info(f"Rank 0: Starting to collect from {host_count-1} nodes")
            all_node_metrics = {f"node_{current_host_rank}": node_metrics}

            for node_rank in range(1, host_count):
                first_rank = node_rank * gpus_per_host
                logger.info(f"Rank 0: Waiting to receive from rank {first_rank}")
                recv_size_tensor = torch.zeros(1, dtype=torch.long)
                if torch.cuda.is_available():
                    recv_size_tensor = recv_size_tensor.to(f"cuda:{local_rank}")
                dist.recv(recv_size_tensor, src=first_rank)
                logger.info(
                    f"Rank 0: Received size {recv_size_tensor.item()} from rank {first_rank}"
                )

                recv_data_tensor = torch.zeros(
                    recv_size_tensor.item(), dtype=torch.uint8
                )
                if torch.cuda.is_available():
                    recv_data_tensor = recv_data_tensor.to(f"cuda:{local_rank}")
                dist.recv(recv_data_tensor, src=first_rank)
                logger.info(f"Rank 0: Received data from rank {first_rank}, unpickling")

                recv_bytes = bytes(recv_data_tensor.cpu().numpy())
                node_data = pickle.loads(recv_bytes)
                all_node_metrics[f"node_{node_rank}"] = node_data
                logger.info(
                    f"Rank 0: Successfully processed metrics from rank {first_rank}"
                )

            # Create and save final metrics
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

    # Final barrier to ensure all ranks wait for metrics to be saved
    if world_size > 1:
        logger.info(f"Rank {rank}: Entering final barrier for metrics saving")
        sys.stdout.flush()
        dist.barrier()
        logger.info(f"Rank {rank}: All ranks synchronized after metrics saving")
        sys.stdout.flush()


def log_health_check_summary(
    node_info,
    summary,
    gpu_metrics,
    system_metrics,
    network_metrics,
    nccl_metrics,
    dcgm_metrics,
    rank,
    world_size,
):
    """Log comprehensive health check summary and results.

    Args:
        node_info: Node information dictionary
        summary: Health check summary
        gpu_metrics: GPU metrics
        system_metrics: System metrics
        network_metrics: Network metrics
        nccl_metrics: NCCL metrics
        dcgm_metrics: DCGM metrics
        rank: Current process rank
        world_size: Total number of processes
    """
    logger.info("=== Pre-flight Check Summary ===")
    logger.info(
        f"Node: {node_info['current_host']} (host rank {node_info['host_rank']}/{node_info['host_count']}, process rank {rank}/{world_size})"
    )
    logger.info(f"Overall Status: {summary['overall_status']}")
    logger.info(f"GPUs detected: {len(gpu_metrics)}")

    # Log detailed GPU information
    if gpu_metrics:
        logger.info("=== GPU Details ===")
        for gpu_id, gpu_data in gpu_metrics.items():
            logger.info(
                f"{gpu_id}: {gpu_data['name']} - {gpu_data['memory_total_mb']}MB total, "
                f"{gpu_data['temperature_c']}°C, {gpu_data['utilization_percent']}% util, "
                f"{gpu_data['power_draw_w']}W"
            )

    logger.info(
        f"System memory: {system_metrics['memory']['available_gb']:.1f}GB available"
    )
    logger.info(f"CPU usage: {system_metrics['cpu']['usage_percent']:.1f}%")
    logger.info(f"EFA available: {network_metrics['efa_available']}")

    # Log EFA performance data if available
    if network_metrics.get("efa_communication_test", {}).get(
        "success"
    ) and network_metrics["efa_communication_test"].get("raw_output"):
        efa_output = network_metrics["efa_communication_test"]["raw_output"]
        if "MB/sec" in efa_output:
            lines = efa_output.split("\n")
            for line in lines:
                if "MB/sec" in line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 6:
                        logger.info(
                            f"EFA bandwidth: {parts[5]} MB/sec ({parts[0]} bytes)"
                        )
                        break

    logger.info(f"NCCL tests available: {nccl_metrics['nccl_tests_available']}")

    # Log NCCL performance data if available
    if nccl_metrics.get("communication_tests"):
        for test_name, test_data in nccl_metrics["communication_tests"].items():
            if test_data.get("success") and test_data.get("results"):
                for result in test_data["results"]:
                    logger.info(
                        f"NCCL {test_name} ({result['size_bytes']} bytes): "
                        f"{result['bandwidth_gbps']:.2f} GB/s, {result['latency_ms']:.2f} ms"
                    )
            elif test_data.get("success") and test_data.get("peak_bandwidth_gbps"):
                logger.info(
                    f"NCCL {test_name} peak bandwidth: {test_data['peak_bandwidth_gbps']:.2f} GB/s"
                )

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
        node_prefix = node_name  # e.g., "node_0", "node_1"

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
    enable_nccl_check=True,
    enable_dcgm_level1=True,
    enable_dcgm_level3=True,
    export_mlflow=False,
    mlflow_uri=None,
    mlflow_experiment_name="sagemaker-health-checks",
):
    """Run comprehensive health checks and collect metrics.

    Performs GPU, system, network, NCCL, and DCGM health checks,
    aggregates results across nodes, and saves to metrics file.

    Args:
        metrics_dir: Directory to save health check metrics
        enable_gpu_check: Enable GPU metrics collection
        enable_system_check: Enable system metrics collection
        enable_network_check: Enable network/EFA checks
        enable_nccl_check: Enable NCCL communication tests
        enable_dcgm_level1: Enable DCGM Level 1 diagnostics
        enable_dcgm_level3: Enable DCGM Level 3 diagnostics

    Returns:
        dict: Complete metrics dictionary
    """
    logger.info("Starting comprehensive pre-flight health checks")

    # Get distributed info from SageMaker environment
    host_count = int(os.environ.get("SM_HOST_COUNT", "1"))
    current_host_rank = int(os.environ.get("SM_CURRENT_HOST_RANK", "0"))
    gpus_per_host = int(os.environ.get("SM_NUM_GPUS", "1"))

    # Calculate distributed parameters from torchrun or SageMaker
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Initialize torch.distributed if using torchrun and not already initialized
    if world_size > 1 and not dist.is_initialized():
        logger.info(f"Initializing torch.distributed for rank {rank}/{world_size}")
        # Use gloo backend if no GPUs, otherwise nccl
        gpu_count = get_gpu_count()
        backend = "nccl" if gpu_count > 0 else "gloo"
        logger.info(f"Using {backend} backend for distributed communication")

        # Set NCCL timeout to 30 minutes to handle long-running DCGM tests
        timeout = timedelta(seconds=1800)
        logger.info(f"Setting distributed timeout to {timeout.total_seconds()} seconds")

        dist.init_process_group(backend=backend, timeout=timeout)
        if gpu_count > 0:
            torch.cuda.set_device(local_rank)

    # Collect all metrics based on enabled checks
    gpu_metrics = extract_gpu_metrics() if enable_gpu_check else {}
    system_metrics = extract_system_metrics() if enable_system_check else {}
    network_metrics = (
        extract_network_metrics(world_size, rank, current_host_rank, gpus_per_host)
        if enable_network_check
        else {}
    )
    nccl_metrics = extract_nccl_metrics(world_size, rank) if enable_nccl_check else {}
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

    # Add node info to metrics
    node_info = {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "current_host": os.environ.get("SM_CURRENT_HOST", "unknown"),
        "host_rank": current_host_rank,
        "host_count": host_count,
        "gpus_per_host": gpus_per_host,
    }

    # Generate summary and recommendations
    summary = generate_summary_and_recommendations(
        gpu_metrics,
        network_metrics,
        nccl_metrics,
        dcgm_metrics,
        world_size,
    )

    # Combine all metrics
    all_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_info": node_info,
        "summary": summary,
        "gpu": gpu_metrics,
        "system": system_metrics,
        "network": network_metrics,
        "nccl": nccl_metrics,
        "dcgm": dcgm_metrics,
    }

    # Save metrics
    save_metrics_to_shared_file(
        all_metrics,
        summary,
        rank,
        world_size,
        current_host_rank,
        host_count,
        gpus_per_host,
        local_rank,
        metrics_dir,
    )

    # Log summary
    log_health_check_summary(
        node_info,
        summary,
        gpu_metrics,
        system_metrics,
        network_metrics,
        nccl_metrics,
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
    """Main execution entry point.

    Parses command-line arguments, runs health checks, and exits.
    """
    parser = argparse.ArgumentParser(
        description="SageMaker Training Cluster Health Checks"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="/opt/ml/output/data/metrics",
        help="Directory to save health check metrics (default: /opt/ml/output/data/metrics)",
    )
    parser.add_argument(
        "--gpu-check",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable GPU metrics collection (default: true)",
    )
    parser.add_argument(
        "--system-check",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable system metrics collection (default: true)",
    )
    parser.add_argument(
        "--network-check",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable network/EFA checks (default: true)",
    )
    parser.add_argument(
        "--nccl-check",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable NCCL communication tests (default: true)",
    )
    parser.add_argument(
        "--dcgm-level1",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable DCGM Level 1 diagnostics (default: true)",
    )
    parser.add_argument(
        "--dcgm-level3",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable DCGM Level 3 diagnostics (default: true)",
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

    logger.info("Starting SageMaker Training Cluster Pre-flight Health Checks")

    run_health_checks(
        metrics_dir=args.metrics_dir,
        enable_gpu_check=args.gpu_check,
        enable_system_check=args.system_check,
        enable_network_check=args.network_check,
        enable_nccl_check=args.nccl_check,
        enable_dcgm_level1=args.dcgm_level1,
        enable_dcgm_level3=args.dcgm_level3,
        export_mlflow=args.export_mlflow,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )

    logger.info("Pre-flight health checks completed successfully")


if __name__ == "__main__":
    main()
