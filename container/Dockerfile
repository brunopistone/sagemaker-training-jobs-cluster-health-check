FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

# EFA installer version
ENV EFA_INSTALLER_VERSION=1.44.0

### Required Packages for the cluster test. DO NOT REMOVE ###
RUN apt-get update && apt-get install -y \
    libfabric-dev \
    datacenter-gpu-manager \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Find and verify NCCL installation
RUN echo "Searching for NCCL..." && \
    find /opt/conda -name "libnccl*" -type f 2>/dev/null | head -5 && \
    find /usr -name "libnccl*" -type f 2>/dev/null | head -5 && \
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || true

# Install NCCL tests - try multiple NCCL locations
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && \
    cd /opt/nccl-tests && \
    PYTHON_VER=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')") && \
    echo "Detected Python version: $PYTHON_VER" && \
    (make NCCL_HOME=/opt/conda/lib/$PYTHON_VER/site-packages/torch || \
    make NCCL_HOME=/usr/local/cuda || \
    make NCCL_HOME=/opt/conda || \
    make) && \
    mkdir -p /usr/local/bin && \
    ln -sf /opt/nccl-tests /usr/local/bin/nccl-tests

# Enable NCCL debugging
ENV NCCL_DEBUG=VERSION

RUN pip install \
    mlflow \
    "psutil>=5.9.0" \
    "sagemaker-mlflow==0.1.0"
