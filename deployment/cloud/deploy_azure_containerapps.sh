#!/usr/bin/env bash
# ==============================================================================
# Deploy to Azure Container Apps (Bonus Requirement)
# ==============================================================================

set -euo pipefail

RESOURCE_GROUP=${AZURE_RG:-"ai-assistant-rg"}
LOCATION=${AZURE_LOCATION:-"eastus"}
ACR_NAME=${AZURE_ACR:-"aiassistantregistry"}
APP_NAME="ai-assistant-app"

echo "Creating Azure Container Registry..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true

echo "Building container image in ACR..."
az acr build --registry "$ACR_NAME" --image "${APP_NAME}:v1" -f deployment/Dockerfile.backend .

echo "Deploying to Azure Container Apps..."
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "ai-assistant-env" \
    --image "${ACR_NAME}.azurecr.io/${APP_NAME}:v1" \
    --target-port 8000 \
    --ingress external \
    --query properties.configuration.ingress.fqdn

echo "Azure Container App deployed."
