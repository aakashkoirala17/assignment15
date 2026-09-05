#!/usr/bin/env bash
# ==============================================================================
# Enterprise AI Assistant - Standalone Docker Run Examples
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Docker Execution Examples for Enterprise AI Assistant"
echo "=========================================================="

# 1. Build Standalone Unified Image
echo "[1] Building standalone image..."
docker build -t ai-assistant:latest -f Dockerfile .

# 2. Run Unified Container with Mock Provider (Port 8000)
echo "[2] Example: Running Backend container with mock provider:"
echo "docker run -d \\"
echo "  --name ai-assistant-backend \\"
echo "  -p 8000:8000 \\"
echo "  -e LLM_PROVIDER=mock \\"
echo "  -e ENABLE_CACHE=true \\"
echo "  -v \$(pwd)/chroma_db:/app/chroma_db \\"
echo "  ai-assistant:latest"

# 3. Run with an environment file
echo ""
echo "[3] Example: Running with .env file:"
echo "docker run -d \\"
echo "  --name ai-assistant-app \\"
echo "  --env-file .env \\"
echo "  -p 8000:8000 \\"
echo "  ai-assistant:latest"

# 4. Build and Run Streamlit Frontend
echo ""
echo "[4] Example: Building and running separate Frontend container:"
echo "docker build -t ai-assistant-frontend:latest -f deployment/Dockerfile.frontend ."
echo "docker run -d \\"
echo "  --name ai-assistant-ui \\"
echo "  -p 8501:8501 \\"
echo "  -e BACKEND_URL=http://host.docker.internal:8000 \\"
echo "  ai-assistant-frontend:latest"

echo ""
echo "=========================================================="
echo " For full multi-container orchestration with Redis, use:"
echo "   docker compose --env-file .env.docker.example up -d --build"
echo "=========================================================="
