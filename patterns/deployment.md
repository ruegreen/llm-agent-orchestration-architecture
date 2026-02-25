# Pattern: Deployment

## Three-Mode Deployment Strategy

A single codebase supports three deployment targets through environment-specific configuration and docker-compose overrides.

### Mode 1: Local Development

Native Ollama installation with Python virtual environment. Optimized for fast iteration.

```bash
# Setup (one-time)
brew install ollama              # macOS
ollama pull llama3.2:3b          # Fast development model
python -m venv agent/.venv
pip install -r agent/requirements.txt

# Run
ollama serve &                   # Background LLM server
uvicorn agent.main:app --reload  # API with hot-reload
python agent/cli.py              # Interactive CLI
```

**Key characteristics:**
- Apple Metal acceleration (M1/M2/M3)
- Sub-second model loading (models stay in memory)
- Hot-reload for agent code changes
- No container overhead

### Mode 2: Container (Podman/Docker)

Multi-service orchestration via docker-compose. Reproducible, isolated environment.

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      OLLAMA_NUM_PARALLEL: 4
      OLLAMA_KEEP_ALIVE: 24h
    networks:
      - llm-network

  agent-api:
    build: ./agent
    ports:
      - "8000:8000"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      PRIMARY_MODEL: llama3.2:3b
    depends_on:
      - ollama
    networks:
      - llm-network

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      - ollama
    networks:
      - llm-network

volumes:
  ollama_data:

networks:
  llm-network:
    driver: bridge
```

### Mode 3: GPU / AI Pod

Docker-compose override adds NVIDIA GPU access, flash attention, and higher parallelism.

```yaml
# docker-compose.gpu.yml (extends base)
services:
  ollama:
    devices:
      - nvidia.com/gpu=all       # CDI device mapping
    environment:
      OLLAMA_NUM_PARALLEL: 16    # Higher parallelism with GPU
      OLLAMA_FLASH_ATTENTION: 1  # ~2x memory efficiency
      OLLAMA_NUM_CTX: 8192       # Larger context window
      OLLAMA_MAX_LOADED_MODELS: 5
      OLLAMA_KEEP_ALIVE: 24h
```

### Mode 4: Kubernetes / OpenShift

Full orchestrated deployment with autoscaling, persistent storage, and external access.

```yaml
# Key Kubernetes resources:
# - Deployment: ollama (1 replica, GPU node selector)
# - Deployment: agent-api (HPA: 1-5 replicas)
# - PVC: ollama-models (model weight storage)
# - Service: ollama (ClusterIP, 11434)
# - Service: agent-api (ClusterIP, 8000)
# - Route/Ingress: external access via ngrok or LoadBalancer
```

## Makefile Interface

A unified command interface abstracts deployment complexity:

```makefile
# Local development
make local-setup    # Install Ollama + Python deps
make local-start    # Start Ollama + API + CLI

# Container
make up             # Start all services (Podman/Docker)
make down           # Stop all services
make logs           # Tail container logs

# GPU / AI Pod
make up-gpu         # Start with GPU overrides
make pull-models    # Download models into container volume

# Kubernetes
make oc-deploy      # Deploy to OpenShift
make oc-status      # Check pod health

# Common
make health         # Check Ollama connectivity
make models         # List available models
make test           # Run agent tests
make clean          # Remove all containers and volumes
```

## Container Build Pattern

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Non-root user for security
RUN useradd -m -s /bin/bash agent
USER agent

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Health check for orchestrators
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment Configuration

```python
# Pydantic BaseSettings — type-safe, environment-driven
class Settings(BaseSettings):
    # Model selection
    primary_model: str = "llama3.2:3b"
    fast_model: str = "llama3.2:3b"

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"

    # Agent parameters
    max_iterations: int = 10
    temperature: float = 0.7

    # Tool toggles
    enable_web_search: bool = True
    enable_file_operations: bool = True
    enable_code_execution: bool = False  # Disabled by default (security)

    # MCP integration
    enable_mcp: bool = True
    mcp_servers: str = ""
    mcp_servers_config: str = ""
    mcp_auth_token: str = ""

    # API
    log_level: str = "INFO"
    require_api_key: bool = False

    class Config:
        env_file = [".env", ".env.local"]
```

## GPU Optimization Settings

| Setting | Local (CPU/Metal) | Container (CPU) | GPU Pod |
|---|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 4 | 4 | 16 |
| `OLLAMA_FLASH_ATTENTION` | 0 | 0 | 1 |
| `OLLAMA_NUM_CTX` | 4096 | 4096 | 8192 |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | 1 | 5 |
| `OLLAMA_KEEP_ALIVE` | 5m | 24h | 24h |
