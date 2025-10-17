# app.py
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from redis import Redis
from rq import Queue
import json
from worker import process_exercises
import re
import time
from typing import List, Dict, Any, Optional
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_N = 10
DEFAULT_LEVEL = 1

# ---------- Deepseek / LLM HTTP config ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
else:
    client = None
    print("ERROR: DEEPSEEK_API_KEY missing!")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "120"))

# -------- FastAPI / Templates --------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redis connection
redis_conn = Redis(host='redis', port=6379, db=0)
q = Queue(connection=redis_conn)

# -------- Routes --------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_n": DEFAULT_N,
        "default_level": DEFAULT_LEVEL
    })

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, n: int = Form(10), level: int = Form(1)):
    n = max(1, min(200, n))
    level = max(1, min(5, level))
    print(f"Generating {n} exercises at level {level}")
    job = q.enqueue(process_exercises, n, level, True)  # Enqueue with html=True
    return templates.TemplateResponse("processing.html", {"request": request, "job_id": job.id})

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result(request: Request, job_id: str):
    job = q.fetch_job(job_id)
    if job.is_finished:
        data = job.result.get("html_data", {})
        return templates.TemplateResponse("exercises.html", {"request": request, **data})
    elif job.is_failed:
        return HTMLResponse(content="<p>Job failed</p>", status_code=500)
    return templates.TemplateResponse("processing.html", {"request": request, "job_id": job_id})

@app.get("/download/{fname}", response_class=FileResponse)
def download(fname: str):
    if ".." in fname or fname.startswith("/"):
        raise HTTPException(status_code=400)
    try:
        return FileResponse(path=fname, media_type="application/json", filename=fname)
    except Exception:
        raise HTTPException(status_code=404)

@app.get("/api/generate/{n}", response_class=JSONResponse)
async def api_generate(n: int, level: Optional[int] = Query(None, ge=1, le=5)):
    n = max(1, min(200, n))
    if level is None:
        level = DEFAULT_LEVEL
    print(f"API: Generating {n} exercises at level {level}")
    job = q.enqueue(process_exercises, n, level, False)  # Enqueue without html
    for _ in range(30):  # Temporary polling for API (can be improved with WebSocket)
        if job.is_finished:
            return JSONResponse(job.result)
        time.sleep(1)
    return JSONResponse({"error": "Task timeout"})

# Simple health endpoint
@app.get("/health")
def health():
    return {"ok": True, "llm": DEEPSEEK_API_KEY is not None}