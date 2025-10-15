# app.py
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
import json
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import requests
import time

# ---------- Deepseek / LLM HTTP config ----------
# Provide DEEPSEEK_API_KEY in Render and GitHub secrets.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/generate")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # tune if needed
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))

# -------- CONFIG --------
DB_PATH = "word_info_level.db"
TABLE = "word_info"
DEFAULT_N = 10
DEFAULT_LEVEL = 1

# -------- FastAPI / Templates --------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------- DB helpers --------
def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_random_rows(conn: sqlite3.Connection, n: int, level: int = None) -> List[sqlite3.Row]:
    cur = conn.cursor()
    # Base query with RANDOM() selection
    q = f"SELECT * FROM {TABLE} "
    params = []
    
    # Add level filter if provided, clamped to 1-5
    if level is not None:
        level = max(1, min(5, level))  # Ensure level is within 1-5
        q += "WHERE level = ? "
        params.append(level)
    q += "ORDER BY RANDOM() LIMIT ?"
    params.append(n)
    
    cur.execute(q, params)
    rows = cur.fetchall()
    return rows

# -------- LLM helpers (batch) --------
def invoke_llm(prompt: str) -> str:
    """
    POST to Deepseek and return a text string.
    Retries on 429/503 with backoff.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "prompt": prompt,
        # tune these as needed:
        "max_tokens": 512,
        "temperature": 0.0,
        # if Deepseek supports streaming or other fields, add them
    }

    attempts = 3
    for attempt in range(attempts):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=DEEPSEEK_TIMEOUT)
            if resp.status_code == 204:
                return ""
            if resp.status_code == 429 or resp.status_code == 503:
                backoff = 2 ** attempt
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = _extract_text_from_response_json(data)
            return text
        except requests.RequestException as e:
            # Retry for transient errors
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Deepseek request failed: {e}")

def _extract_text_from_response_json(resp_json: dict) -> str:
    """
    Try various possible response shapes and return the textual output.
    Adjust if Deepseek uses a different schema.
    """
    if not isinstance(resp_json, dict):
        return str(resp_json)
    # common patterns:
    if "output" in resp_json and isinstance(resp_json["output"], str):
        return resp_json["output"]
    if "text" in resp_json and isinstance(resp_json["text"], str):
        return resp_json["text"]
    if "response" in resp_json and isinstance(resp_json["response"], str):
        return resp_json["response"]
    # choices -> text
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for k in ("text", "message", "content", "output"):
                if k in first and isinstance(first[k], str):
                    return first[k]
            # sometimes message:{content: "..."}
            if "message" in first and isinstance(first["message"], dict):
                cont = first["message"].get("content")
                if isinstance(cont, str):
                    return cont
    # fallback to JSON string if nothing matched
    try:
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception:
        return str(resp_json)

def build_definition_prompt(words: List[str]) -> str:
    words_json = json.dumps(words, ensure_ascii=False)
    p = (
        "Return ONLY a JSON array in the SAME ORDER as INPUT_WORDS.\n"
        "For each word provide a concise one-sentence learner-friendly definition (6-20 words).\n"
        'Format: [{"word":"...","definition":"..."}]\n\n'
        f"INPUT_WORDS: {words_json}\n\nReturn the JSON array only."
    )
    return p

def build_sentence_prompt(words: List[str]) -> str:
    words_json = json.dumps(words, ensure_ascii=False)
    p = (
        "Return ONLY a JSON array in the SAME ORDER as INPUT_WORDS.\n"
        "For each word produce a single natural sentence (8-18 words) with the target word replaced by a blank token '____'.\n"
        "Do NOT include the target word in the sentence.\n"
        'Format: [{"word":"...","sentence":"..."}]\n\n'
        f"INPUT_WORDS: {words_json}\n\nReturn the JSON array only."
    )
    return p

def generate_definitions(words: List[str]) -> Dict[str, str]:
    if not words:
        return {}
    try:
        prompt = build_definition_prompt(words)
        raw = invoke_llm(prompt)
        arr = _extract_text_from_response_json(raw)
        out = {}
        if arr:
            for item in arr:
                w = item.get("word")
                d = (item.get("definition") or "").strip()
                out[w] = d
            return out
    except Exception:
        pass
    # fallback: short per-word prompt (using DeepSeek)
    out = {}
    for w in words:
        try:
            p = f'Return a single short learner-friendly definition (6-20 words) for the word "{w}". Return only the definition.'
            raw = invoke_llm(p)
            out[w] = raw.strip().splitlines()[0]
        except Exception:
            out[w] = ""
    return out

def generate_sentences(words: List[str]) -> Dict[str, str]:
    if not words:
        return {}
    try:
        prompt = build_sentence_prompt(words)
        raw = invoke_llm(prompt)
        arr = _extract_text_from_response_json(raw)
        out = {}
        if arr:
            for item in arr:
                w = item.get("word")
                s = (item.get("sentence") or "").strip()
                out[w] = s
            return out
    except Exception:
        pass
    out = {}
    for w in words:
        try:
            p = f'Write one natural sentence (8-18 words) that would contain the word "{w}", but replace the word with "____". Return only the sentence.'
            raw = invoke_llm(p)
            out[w] = raw.strip().splitlines()[0]
        except Exception:
            out[w] = ""
    return out

# -------- Exercise builder --------
def build_exercises_from_rows(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    words = [r["word"] for r in rows]
    # DB-provided definitions / examples
    db_defs = {r["word"]: r["definition"].strip() for r in rows if "definition" in r.keys() and r["definition"]}
    db_examples = {r["word"]: r["example"].strip() for r in rows if "example" in r.keys() and r["example"]}

    # generate definitions for missing
    need_def = [w for w in words if w not in db_defs]
    gen_defs = generate_definitions(need_def) if need_def else {}
    defs = {**db_defs, **gen_defs}

    # build sentence blanks: try to replace the word in DB example if possible
    db_blanks = {}
    for w, ex in db_examples.items():
        # naive replacement, case-insensitive
        if re.search(re.escape(w), ex, flags=re.IGNORECASE):
            s_blank = re.sub(re.escape(w), "____", ex, count=1, flags=re.IGNORECASE)
            db_blanks[w] = s_blank

    need_sent = [w for w in words if w not in db_blanks]
    gen_sents = generate_sentences(need_sent) if need_sent else {}
    sents = {**db_blanks, **gen_sents}

    # Build exercises
        # --- Part A (definitions) ---
    part_a = []
    for w in words:
        part_a.append({
            "part": "A",
            "type": "definition_fill",
            "prompt": defs.get(w, "(no definition available)"),
            "answer": w
        })

    # --- Part B (sentences) ---
    part_b = []
    for w in words:
        part_b.append({
            "part": "B",
            "type": "sentence_fill",
            "prompt": sents.get(w, "(no sentence available)"),
            "answer": w
        })

    # shuffle each part independently so order differs
    random.shuffle(part_a)
    random.shuffle(part_b)

    return {
        "chosen_words": words,  # keep word bank in DB order
        "part_a": part_a,
        "part_b": part_b,
    }


# -------- Routes --------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Pass default_level for template select pre-selection
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_n": DEFAULT_N,
        "default_level": DEFAULT_LEVEL  # e.g., 1 for "Beginner"
    })

@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, n: int = Form(10), level: int = Form(1)):
    # Clamp n and level
    n = max(1, min(200, int(n)))
    level = max(1, min(5, level))  # Clamp level to 1-5
    # Add debug print (visible in Render logs)
    print(f"Generating {n} exercises at level {level}")  # Check logs after submit
    conn = get_connection()
    
    try:
        rows = get_random_rows(conn, n, level=level)
        if not rows:
            raise HTTPException(status_code=500, detail="No words found for this level.")
        payload = build_exercises_from_rows(rows)
        # Map level int to string for display in template (e.g., "Level 1: Beginner")
        level_str = {1: "Beginner", 2: "Elementary", 3: "Intermediate", 4: "Advanced", 5: "Expert"}.get(level, "Unknown")
        import uuid  # Add at top if needed
        json_filename = f"exercises_{uuid.uuid4().hex[:8]}.json"  # Unique name
# Save JSON for download (optional, but enables the link)
        with open(json_filename, 'w') as f:
            json.dump({
                "exercises": payload,
                "level": level,
                "level_str": level_str,
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        return templates.TemplateResponse("exercises.html", {
            "request": request,
            "payload": payload,
            "level": level,
            "level_str": level_str, # For nicer display
            "json_filename": json_filename
        })
    finally:
        conn.close()

@app.get("/download/{fname}", response_class=FileResponse)
def download(fname: str):
    # basic safety check
    if ".." in fname or fname.startswith("/"):
        raise HTTPException(status_code=400)
    try:
        return FileResponse(path=fname, media_type="application/json", filename=fname)
    except Exception:
        raise HTTPException(status_code=404)

@app.get("/api/generate/{n}", response_class=JSONResponse)
def api_generate(n: int, level: Optional[int] = Query(None, ge=1, le=5, description="Difficulty level (1-5)")):
    n = max(1, min(200, n))
    # Default to DEFAULT_LEVEL if not provided
    if level is None:
        level = DEFAULT_LEVEL
    # Add debug print for API calls too
    print(f"API: Generating {n} exercises at level {level}")
    conn = get_connection()
    try:
        rows = get_random_rows(conn, n, level=level)
        if not rows:
            return JSONResponse({"error": "No words found for the given level or none available."})
        payload = build_exercises_from_rows(rows)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC (Custom Format)")
        response_data = {
            "exercises": payload,
            "timestamp": timestamp,
            "level": level  # Include level in response
        }
        return JSONResponse(response_data)
    finally:
        conn.close()

# Simple health endpoint
@app.get("/health")
def health():
    return {"ok": True, "llm": DEEPSEEK_API_KEY is not None}