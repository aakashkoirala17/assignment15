# Enterprise AI Assistant: Deployment Guide

This guide covers deployment options for the AI Assistant across local, containerized, and major cloud environments.

---

## 1. Local Development (Virtual Environment)

### Prerequisites:
- Python 3.12+
- Docker (optional for container testing)

### Steps:
```bash
# 1. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Configure environment variables (.env)
cp .env.example .env

# Customize provider or settings if desired:
# edit .env


# 4. Start FastAPI Backend (Port 8000)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. In a separate terminal, start Streamlit Frontend (Port 8501)
streamlit run frontend/app.py --server.port 8501
```
Access points:
- Streamlit Web UI: `http://localhost:8501`
- FastAPI Interactive Swagger Docs: `http://localhost:8000/docs`
- Healthcheck Endpoint: `http://localhost:8000/healthz`

---

## 2. Docker Single Container Deployment

To package and run the application as a standalone container:
```bash
# Build image
docker build -t ai-assistant:latest .

# Run container
docker run -d \
    --name ai-assistant \
    -p 8000:8000 \
    -e LLM_PROVIDER=mock \
    -e ENABLE_CACHE=true \
    ai-assistant:latest

# Verify health
curl http://localhost:8000/healthz
```

---

## 3. Docker Compose Multi-Container Orchestration

Orchestrates the FastAPI backend, Streamlit frontend, and Redis cache:
```bash
# Launch the complete stack
docker compose up -d --build

# View container status
docker compose ps

# Follow logs
docker compose logs -f

# Teardown stack
docker compose down
```

Service mapping:
| Service | Internal URL | External Port | Role |
| :--- | :--- | :--- | :--- |
| `backend` | `http://backend:8000` | `8000` | FastAPI REST & SSE API |
| `frontend` | `http://frontend:8501`| `8501` | Streamlit User Interface |
| `redis` | `redis://redis:6379` | `6379` | Cache & Rate Limiter Store |

---

## 4. Kubernetes Deployment

To deploy onto Minikube, GKE, EKS, or AKS:
```bash
# Apply deployment and service manifests
kubectl apply -f deployment/k8s/app-deployment.yaml

# Check rollout status
kubectl rollout status deployment/ai-assistant-backend
kubectl rollout status deployment/ai-assistant-frontend

# Retrieve external LoadBalancer IP
kubectl get svc ai-assistant-frontend-service
```

---

## 5. Major Cloud Deployments (Bonus)

### Google Cloud Run:
```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
bash deployment/cloud/deploy_gcp_cloudrun.sh
```

### AWS ECS / App Runner:
```bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="123456789012"
bash deployment/cloud/deploy_aws_ecs.sh
```

### Azure Container Apps:
```bash
export AZURE_RG="ai-assistant-rg"
export AZURE_LOCATION="eastus"
bash deployment/cloud/deploy_azure_containerapps.sh
```
