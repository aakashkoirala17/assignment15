#!/usr/bin/env bash
# ==============================================================================
# Deploy to Google Cloud Run (Bonus Requirement)
# ==============================================================================

set -euo pipefail

PROJECT_ID=${GCP_PROJECT_ID:-"your-gcp-project-id"}
REGION=${GCP_REGION:-"us-central1"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/ai-assistant-backend:v1"

echo "Building container for Google Cloud Run..."
gcloud builds submit --tag "$IMAGE_NAME" -f deployment/Dockerfile.backend .

echo "Deploying service to Cloud Run with autoscaling (0 to 10 instances)..."
gcloud run deploy ai-assistant-backend \
    --image "$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars LLM_PROVIDER=gemini,RATE_LIMIT_REQUESTS_PER_MINUTE=120,ENABLE_CACHE=true

echo "Cloud Run deployment complete!"
