#!/usr/bin/env bash
# ==============================================================================
# Deploy to AWS ECS Fargate / App Runner (Bonus Requirement)
# ==============================================================================

set -euo pipefail

AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"123456789012"}
REPO_NAME="ai-assistant-backend"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "Authenticating with AWS ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building and tagging image..."
docker build -t "$IMAGE_URI" -f deployment/Dockerfile.backend .

echo "Pushing image to ECR..."
docker push "$IMAGE_URI"

echo "Deploying to AWS App Runner service..."
aws apprunner create-service \
    --service-name ai-assistant-backend \
    --source-configuration "ImageRepository={ImageIdentifier=${IMAGE_URI},ImageRepositoryType=ECR,ImageConfiguration={Port=8000}}" \
    --instance-configuration "Cpu=1024,Memory=2048"

echo "AWS deployment initiated successfully."
