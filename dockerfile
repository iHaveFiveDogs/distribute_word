FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Use Render's dynamic port or fallback to 8001 for local dev
EXPOSE 10000

# Run uvicorn on the dynamic port; RQ worker runs in a separate service, not here
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
