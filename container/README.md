# Container images

Two Dockerfiles are provided for building the cluster health-check image:

- **`Dockerfile`** — extends the AWS Deep Learning Container
  (`pytorch-training` SageMaker image). Recommended default: it already ships
  PyTorch, CUDA, NCCL, and EFA, so it only adds the health-check dependencies
  (`libfabric-dev`, DCGM, `nccl-tests`, `mlflow`, `psutil`).
- **`Dockerfile.fromscratch`** — builds from `nvidia/cuda` and installs Python
  3.12, PyTorch, EFA, the AWS OFI NCCL plugin, `nccl-tests`, DCGM, MPI, and the
  SageMaker training toolkit from scratch. Use when you cannot base off the AWS
  DLC.

## Build and push

`create-image.sh` builds the image, creates the ECR repository if needed, logs
in, and pushes:

```bash
./create-image.sh <REPO_NAME> [TAG] [DOCKERFILE]

# Examples
./create-image.sh sagemaker-cluster-test                       # Dockerfile, tag latest
./create-image.sh sagemaker-cc-cluster-test latest Dockerfile.fromscratch
```

The script auto-detects the AWS region and account, and uses
`--network sagemaker` when run inside SageMaker Studio.
