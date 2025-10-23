# Distribute Word – Vocabulary Exercise Generator

A FastAPI-based web application to generate vocabulary exercises using a language model (DeepSeek/OpenAI compatible) and a local word database. Supports distributed processing via Redis and RQ, containerization with Docker, and Kubernetes/Helm deployment.

---

## Features

- **Generate Vocabulary Exercises:**
  - Web UI and API to generate customizable English vocabulary exercises.
  - Exercises include definitions and fill-in-the-blank sentences, powered by an LLM.
- **Asynchronous Processing:**
  - Uses Redis and RQ for background job queuing and status tracking.
- **Prometheus Monitoring:**
  - Exposes `/metrics` endpoint for Prometheus scraping.
- **Containerized & Cloud-ready:**
  - Docker Compose setup for local development.
  - Kubernetes manifests and Helm chart for cloud deployment.

---

## Quick Start (Local)

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized setup)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
- Copy `.env` and update `DEEPSEEK_API_KEY` and `REDIS_URL` as needed.

### 3. Start Redis (Docker recommended)
```bash
docker-compose up redis
```

### 4. Run the web app
```bash
uvicorn app:app --reload --port 8001
```

### 5. Run the worker
```bash
rq worker --url $REDIS_URL
```

### 6. Access the app
- Web UI: [http://localhost:8001](http://localhost:8001)
- API: `/api/generate/{n}`
- Health: `/health`
- Metrics: `/metrics`

---

## Docker Compose

To run all services (web, worker, Redis, Prometheus) locally:
```bash
docker-compose up --build
```

---

## Kubernetes & Helm

- Kubernetes manifests are in the `k8s/` folder.
- Helm chart is in `distribute-word-k8-template/`.

---

## File Structure

```
├── app.py                  # FastAPI app (web server)
├── worker.py               # RQ worker logic
├── requirements.txt        # Python dependencies
├── docker-compose.yaml     # Multi-service Docker config
├── dockerfile              # Web/worker Docker image
├── .env                    # Environment variables
├── word_info_level.db      # SQLite word database
├── static/                 # CSS and assets
├── templates/              # Jinja2 HTML templates
├── k8s/                    # Kubernetes manifests
├── distribute-word-k8-template/ # Helm chart
├── prometheus.yml          # Prometheus config
├── test_app.py             # Pytest-based tests
```

---

## Configuration

- **Environment Variables:**
  - `DEEPSEEK_API_KEY` – API key for DeepSeek/OpenAI LLM.
  - `REDIS_URL` – Redis connection string (e.g., `redis://redis:6379`).
  - See `.env` for all options.

---

## Testing

```bash
pytest test_app.py
```

---

## License

MIT
