#!/bin/bash
set -euo pipefail

# Function to detect if running in SageMaker Studio
is_sagemaker_studio() {
    # Check for SageMaker-specific environment variables (use :- so that
    # referencing them while unset does not trip `set -u`).
    if [[ -n "${SM_CURRENT_HOST:-}" ]] || [[ -n "${SAGEMAKER_INTERNAL_IMAGE_URI:-}" ]] || [[ -n "${SM_USER_ID:-}" ]]; then
        return 0  # True - in SageMaker Studio
    fi

    # Check for SageMaker Studio specific paths
    if [[ -d "/opt/ml" ]] && [[ -f "/opt/ml/metadata/resource-metadata.json" ]]; then
        return 0  # True - in SageMaker Studio
    fi

    # Check if running in a container with SageMaker characteristics
    if [[ -f "/.dockerenv" ]] && [[ $(hostname) =~ ^sagemaker-* ]]; then
        return 0  # True - likely in SageMaker Studio
    fi

    # Check for SageMaker Studio user
    if [[ $(whoami) == "sagemaker-user" ]]; then
        return 0  # True - likely in SageMaker Studio
    fi

    return 1  # False - not in SageMaker Studio
}

# Check if required parameters are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <REPO_NAME> [TAG] [DOCKERFILE]"
    echo "Example: $0 my-xgboost-image"
    echo "Example: $0 my-xgboost-image v1.0.0"
    echo "Example: $0 my-xgboost-image v1.0.0 Dockerfile.fromscratch"
    exit 1
fi

# Check required tools up front
for tool in aws docker; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: required tool '$tool' is not installed or not on PATH."
        exit 1
    fi
done

# Try to get region from STS ARN first, fallback to configured region.
# The STS call may fail (e.g. no credentials); `|| true` keeps `set -e`/pipefail
# from aborting before the fallback below runs.
AWS_REGION=$(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null | cut -d':' -f4 || true)
if [ -z "$AWS_REGION" ]; then
    AWS_REGION=$(aws configure get region || true)
fi

# If still empty, exit with error
if [ -z "$AWS_REGION" ]; then
    echo "Error: Could not determine AWS region. Please configure your AWS CLI or set AWS_DEFAULT_REGION environment variable."
    exit 1
fi
export AWS_REGION

# Resolve the account id (separate from `export` so a failure is not masked by
# the export builtin's exit status, and validate it is non-empty).
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "None" ]; then
    echo "Error: Could not determine AWS account id. Check your AWS credentials."
    exit 1
fi
export ACCOUNT

export REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/"
export REPO_NAME="$1"
export TAG="${2:-latest}"          # Use provided tag or default to 'latest'
export DOCKERFILE="${3:-Dockerfile}"  # Use provided dockerfile or default to 'Dockerfile'

IMAGE_URI="${REGISTRY}${REPO_NAME}:${TAG}"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile '$DOCKERFILE' not found in $(pwd)."
    exit 1
fi

echo "This process may take 10-15 minutes to complete..."

echo "Building image using ${DOCKERFILE}..."

# Detect environment and use appropriate Docker build command. A failed build
# now aborts the script (set -e) instead of proceeding to push a stale image.
if is_sagemaker_studio; then
    echo "Detected SageMaker Studio environment - using --network sagemaker"
    docker build --network sagemaker --platform linux/amd64 -f "$DOCKERFILE" -t "$IMAGE_URI" .
else
    echo "Detected local/standard environment - using default network"
    docker build --platform linux/amd64 -f "$DOCKERFILE" -t "$IMAGE_URI" .
fi

# Create repository if needed. Use the command directly as the `if` condition so
# a non-zero exit (repo does not exist) does not trip `set -e`.
echo "Checking if repository exists..."
if aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Repository ${REPO_NAME} already exists"
else
    echo "Creating repository ${REPO_NAME}..."
    aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"
fi

# Login to registry
echo "Logging in to ${REGISTRY} ..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"

echo "Pushing image to ${IMAGE_URI} ..."

# Push image to registry
docker image push "$IMAGE_URI"

echo "Image push completed successfully!"
