# app.py
from fastapi import FastAPI, Request, Form, HTTPException ,Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
import json
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# ChatOllama import (same as your environment)
from langchain_community.chat_models import ChatOllama

# -------- CONFIG --------
DB_PATH = "word_info_level.db"
TABLE = "word_info"
DEFAULT_N = 10

CHAT_MODEL = "mistral:7b"
CHAT_BASE_URL = "http://localhost:11434"
CHAT_TEMPERATURE = 0.0

# instantiate chat model
try:
    chat = ChatOllama(model=CHAT_MODEL, base_url=CHAT_BASE_URL, temperature=CHAT_TEMPERATURE)
except Exception as e:
    # we'll lazily handle LLM failures later
    chat = None
    print("Warning: ChatOllama client couldn't be instantiated at import time:", e)

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
    if chat is None:
        raise RuntimeError("LLM client not available")
    resp = chat.invoke(prompt)
    # langchain_community wrapper sometimes returns object with .content
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)

def extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    t = text.strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        # find first [ and last ]
        first = t.find('[')
        last = t.rfind(']')
        if first != -1 and last != -1 and last > first:
            try:
                cand = t[first:last+1]
                parsed = json.loads(cand)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return None

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
        arr = extract_json_array(raw)
        out = {}
        if arr:
            for item in arr:
                w = item.get("word")
                d = (item.get("definition") or "").strip()
                out[w] = d
            return out
    except Exception:
        pass
    # fallback: short per-word prompt (deterministic)
    out = {}
    for w in words:
        try:
            p = f'Return a single short learner-friendly definition (6-20 words) for the word "{w}". Return only the definition.'
            raw = invoke_llm(p) if chat else f"(no LLM) definition for {w}"
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
        arr = extract_json_array(raw)
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
            raw = invoke_llm(p) if chat else f"(no LLM) Example sentence with ____ for {w}"
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
    return templates.TemplateResponse("index.html", {"request": request, "default_n": DEFAULT_N})

@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, n: int = Form(10), level: int = Form(1)):
    # Clamp n and level
    n = max(1, min(200, int(n)))
    level = max(1, min(5, level))  # Clamp level to 1-5
    conn = get_connection()
    try:
        rows = get_random_rows(conn, n, level=level)
        if not rows:
            raise HTTPException(status_code=500, detail="No words found for this level.")
        payload = build_exercises_from_rows(rows)
        return templates.TemplateResponse("exercises.html", {"request": request, "payload": payload, "level": level})
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
    conn = get_connection()
    try:
        rows = get_random_rows(conn, n, level=level)
        if not rows:
            return JSONResponse({"error": "No words found for the given level or none available."})
        payload = build_exercises_from_rows(rows)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC (Custom Format)")
        response_data = {"exercises": payload, "timestamp": timestamp}
        return JSONResponse(response_data)
    finally:
        conn.close()

# Simple health endpoint
@app.get("/health")
def health():
    return {"ok": True, "llm": chat is not None}
