# SageMaker Training Jobs Cluster Health Checks

**Comprehensive pre-flight validation for SageMaker multi-GPU training clusters to prevent costly training failures.**

## What This Does

This tool runs **before** your actual training job to validate that your SageMaker cluster is properly configured and ready for distributed GPU training. It performs comprehensive health checks across:

- **GPU Hardware**: Memory, temperature, drivers, utilization
- **Network Fabric**: EFA (Elastic Fabric Adapter) connectivity for multi-node communication
- **Inter-GPU Communication**: NCCL performance testing across nodes and GPUs
- **System Resources**: CPU, memory, disk availability
- **Hardware Diagnostics**: DCGM validation for GPU health

## Why This Is Important

**Training jobs on large GPU clusters are expensive and time-sensitive.** Common issues that cause training failures:

- **Network misconfiguration**: EFA not properly configured, causing communication timeouts
- **GPU hardware issues**: Memory errors, thermal throttling, driver problems
- **NCCL communication failures**: Poor bandwidth between nodes, blocking collective operations
- **Resource constraints**: Insufficient memory, disk space, or CPU resources

**Cost Impact**: A failed 8-GPU training job can waste hundreds of dollars and hours of compute time. This tool catches issues in minutes, not hours.

## Where To Use

### **Pre-Training Validation**

- Run before expensive multi-GPU training jobs
- Validate new cluster configurations
- Test after infrastructure changes

### **Cluster Commissioning**

- Validate new SageMaker training environments
- Verify EFA network setup for multi-node training
- Baseline performance testing for GPU clusters

### **Troubleshooting**

- Diagnose training job failures
- Identify performance bottlenecks
- Validate fixes after infrastructure changes

### **CI/CD Pipelines**

- Automated cluster validation before training workflows
- Infrastructure testing in MLOps pipelines
- Performance regression detection

## Use Cases by Scale

| Scenario        | Instance Count       | Primary Validation                            |
| --------------- | -------------------- | --------------------------------------------- |
| **Single GPU**  | 1 × ml.g5.xlarge     | GPU health, basic diagnostics                 |
| **Multi-GPU**   | 1 × ml.g5.12xlarge   | GPU communication, NCCL performance           |
| **Multi-Node**  | 2+ × ml.p4d.24xlarge | Inter-node networking, EFA validation         |
| **Large Scale** | 4+ × ml.p5.48xlarge  | Full cluster communication, bandwidth testing |

## Health Checks Performed

### **GPU Health Validation**

- **Hardware Detection**: GPU count, model identification (A10G, V100, etc.)
- **Memory Analysis**: Total, used, free memory per GPU
- **Thermal Monitoring**: Temperature readings and thermal throttling detection
- **Power Management**: Power draw monitoring and efficiency validation
- **Driver Validation**: NVIDIA driver version compatibility
- **Utilization Tracking**: GPU usage percentage and availability

### **System Resource Validation**

- **Memory Analysis**: RAM availability, usage patterns, memory pressure
- **Disk Space**: Storage capacity, free space, I/O performance readiness
- **CPU Monitoring**: Core utilization, availability for training processes
- **Resource Sufficiency**: Validates adequate resources for distributed training

### **Network Fabric Testing**

- **EFA Detection**: Elastic Fabric Adapter availability and configuration
- **Provider Validation**: Network provider enumeration and health
- **Connectivity Testing**: Inter-node communication pathway validation
- **Bandwidth Readiness**: Network fabric performance for multi-node training

### **NCCL Communication Performance**

- **All-Reduce Testing**: Collective operation bandwidth across GPUs/nodes
- **All-Gather Validation**: Data gathering performance and reliability
- **Dynamic Scaling**: Test parameters adapt to cluster size (1-8+ GPUs)
- **Performance Metrics**: Bandwidth measurements (GB/s) with thresholds
- **Timeout Management**: Appropriate limits for single vs multi-node setups

### **DCGM Hardware Diagnostics**

- **Basic Validation**: Driver, NVML, CUDA library compatibility
- **Extended Testing**: PCIe connectivity, GPU memory integrity
- **Stress Testing**: Targeted power and memory bandwidth validation
- **Deployment Checks**: Permissions, environment variables, persistence mode
- **Hardware Health**: Comprehensive GPU subsystem validation

### **Cluster-Level Analysis**

- **Topology Detection**: World size, node count, GPUs per node mapping
- **Multi-node Identification**: Distributed environment configuration
- **Communication Patterns**: Inter-node vs intra-node performance testing
- **Performance Baselines**: Bandwidth thresholds and optimization recommendations

### **Intelligent Reporting**

- **Overall Status**: PASS/WARN/FAIL with clear reasoning
- **Issue Identification**: Specific problems with root cause analysis
- **Actionable Recommendations**: Performance tuning and configuration guidance
- **Cost-Benefit Analysis**: Training readiness vs investigation requirements

## Quick Start

1. **Login to ECR (if using AWS SageMaker base image)**

   ```bash
   aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <IMAGE_URI>
   ```

2. **Build and push container**

   ```bash
   cd container
   ./create-image.sh sagemaker-cluster-test latest
   ```

3. **Run health checks**
   - Open `notebook.ipynb`
   - Update networking configuration (subnet/security group IDs)
   - Execute cells to launch training job

## What It Checks

- **GPU**: Memory, temperature, utilization, driver version
- **System**: CPU, memory, disk usage
- **Network**: EFA availability and fabric info
- **Communication**: NCCL all_reduce/all_gather tests
- **Hardware**: DCGM diagnostics

## Configuration

Update in notebook:

```python
# Instance configuration
instance_type = "ml.g5.12xlarge"
instance_count = 2

# Networking (required)
networking = Networking(
    subnet_ids=["subnet-xxxxxxxxx"],
    security_group_ids=["sg-xxxxxxxxx"],
)
```

## Distributed Configuration Scenarios

| Scenario                    | Instance Count | Torchrun Required | Configuration                                                                                                     |
| --------------------------- | -------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Single node, single GPU** | 1              | No                | Remove `distributed=Torchrun()`                                                                                   |
| **Single node, multi-GPU**  | 1              | Optional          | Use `distributed=Torchrun()` or `distributed=MPI(process_count_per_node=<NUM_GPUS>)` for GPU communication tests  |
| **Multi-node, single GPU**  | >1             | Yes               | Use `distributed=Torchrun()` or `distributed=MPI(process_count_per_node=<NUM_GPUS>)` for inter-node communication |
| **Multi-node, multi-GPU**   | >1             | Yes               | Use `distributed=Torchrun()` or `distributed=MPI(process_count_per_node=<NUM_GPUS>)` for full cluster validation  |

**When to use Torchrun:**

- **Required**: Multi-node setups (validates inter-node communication)
- **Recommended**: Multi-GPU scenarios (tests NCCL across GPUs)
- **Optional**: Single GPU (basic health checks only)

**Example configurations:**

```python
# Single node health check
model_trainer = ModelTrainer(
    # ... other config
    # distributed=Torchrun(),  # Remove this line
)

# Multi-node cluster validation
model_trainer = ModelTrainer(
    # ... other config
    distributed=Torchrun(),  # Keep this line
)
```

## Output

Health check metrics saved to:

- S3: `s3://bucket/job-name/metrics/health_check_metrics.json`
- CloudWatch Logs: Training job logs

## Dockerfile Customization

The Dockerfile has two sections:

**Customizable** (can be changed):

- `FROM` base image
- Any additional packages before the required section

**Required** (DO NOT REMOVE):

```dockerfile
### Required Packages for the cluster test. DO NOT REMOVE ###
RUN apt-get update && apt-get install -y \
    libfabric-dev \
    datacenter-gpu-manager \
    && rm -rf /var/lib/apt/lists/*

# Install NCCL tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \
    cd /opt/nccl-tests && \
    make && \
    ln -s /opt/nccl-tests /usr/local/cuda/efa/test-

RUN pip install "psutil>=5.9.0"
```

These packages enable:

- `libfabric-dev`: EFA network fabric support
- `datacenter-gpu-manager`: GPU diagnostics via DCGM
- `nccl-tests`: Multi-GPU communication validation
- `psutil`: System resource monitoring

## Requirements

- AWS CLI configured
- Docker installed
- SageMaker execution role with ECR permissions
- VPC with EFA-enabled subnets for multi-node jobs
