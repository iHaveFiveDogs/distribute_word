# worker.py
import random
import sqlite3
from typing import List, Dict, Any, Optional
from redis import Redis
from rq import Queue
import sqlite3
from typing import List, Dict
import json
import re
from datetime import datetime
from openai import OpenAI
import os
import requests
import time
from dotenv import load_dotenv
from urllib.parse import urlparse
load_dotenv()
# Redis connection
# Load REDIS_URL from environment
def get_redis_connection():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("REDIS_URL environment variable is not set")
    parsed_url = urlparse(redis_url)
    return Redis(
        host=parsed_url.hostname,
        port=parsed_url.port,
        password=parsed_url.password if parsed_url.password else None,
        ssl=(parsed_url.scheme == "rediss"),
        ssl_cert_reqs=None if parsed_url.scheme == "rediss" else None,
        decode_responses=True
    )


# Reuse your LLM and exercise logic
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "120"))

DB_PATH = "word_info_level.db"
TABLE = "word_info"


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

# Copy invoke_llm, generate_definitions, generate_sentences, build_exercises_from_rows here
# (Paste the functions from app.py)
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
def invoke_llm(prompt: str, system_prompt: str = "You are a helpful assistant that generates concise English language content.") -> str:
    """
    Use OpenAI SDK for DeepSeek chat completions. Returns text or empty on fail.
    """
    if not client:
        print("ERROR: No DeepSeek client (missing key)")
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        print(f"DeepSeek: Sending {len(prompt)} char prompt...")  # Log start
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=256,  # Reduced for speed
            temperature=0.0,
            timeout=DEEPSEEK_TIMEOUT,  # NEW: SDK timeout
            stream=False
        )
        text = response.choices[0].message.content.strip()
        print(f"DeepSeek success: {len(text)} chars returned")  # Log output
        return text
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return ""  # Graceful fallback

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
        'In definitions, avoid using the target word or its root form.\n'
        f"INPUT_WORDS: {words_json}\n\nReturn the JSON array only."
    )
    return p

def build_sentence_prompt(words: List[str]) -> str:
    words_json = json.dumps(words, ensure_ascii=False)
    return (
        "Return ONLY a valid JSON array in the SAME ORDER as INPUT_WORDS.\n"
        "For each word, produce a single natural sentence (8-18 words) with the target word replaced by '____'.\n"
        "Do NOT include the target word in the sentence.\n"
        'Format: [{"word":"...","sentence":"..."}]\n\n'
        f"INPUT_WORDS: {words_json}\n\nReturn the JSON array only, no extra text."
    )

def generate_definitions(words: List[str]) -> Dict[str, str]:
    if not words:
        return {}
    out = {}
    # Batch if small
    if len(words) <= 3:  # Avoid overload
        try:
            prompt = build_definition_prompt(words)
            raw = invoke_llm(prompt)
            if raw:
                # Clean & parse JSON
                cleaned = re.sub(r'```json\s*|\s*```', '', raw.strip())
                arr = json.loads(cleaned)
                if isinstance(arr, list):
                    for item in arr:
                        w = item.get("word")
                        d = item.get("definition", "").strip()
                        if w and d:
                            out[w] = d
            print(f"Batch defs parsed: {len(out)} items")
        except Exception as e:
            print(f"Batch defs error: {e} - per-word fallback")
    
    # Per-word fallback (always run for missing)
    for w in words:
        if w not in out:
            try:
                p = f'Return a single short learner-friendly definition (6-20 words) for the word "{w}". Return only the definition, no extra text.'
                raw = invoke_llm(p)
                if raw:
                    cleaned = re.sub(r'```.*```', '', raw.strip(), flags=re.DOTALL).strip()
                    out[w] = cleaned.splitlines()[0]
                else:
                    out[w] = f"A {w.lower()} is a basic concept in English vocabulary. (fallback)"
            except Exception as e:
                print(f"Per-word def fail for {w}: {e}")
                out[w] = f"Definition for {w} (error fallback)"
            if not out.get(w):
                # Word-specific fallback (simple rules)
                if w.lower() in ['compose', 'defuse', 'antebellum']:  # Add your words/DB samples
                    out[w] = {
                        'compose': 'To compose is to create or produce something, like music or writing.',
                        'defuse': 'To defuse is to make a dangerous situation less tense or harmful.',
                        'antebellum': 'Antebellum refers to the period before a war, especially the U.S. Civil War.'
                    }.get(w.lower(), f"A {w.lower()} is a fundamental word in English learning. (fallback)")
                else:
                    out[w] = f"A {w.lower()} is a key vocabulary term for intermediate learners. (fallback)"
    return out

def generate_sentences(words: List[str]) -> Dict[str, str]:
    if not words:
        return {}
    out = {}
    if len(words) <= 3:
        try:
            prompt = build_sentence_prompt(words)
            raw = invoke_llm(prompt)
            if raw:
                cleaned = re.sub(r'```json\s*|\s*```', '', raw.strip())
                arr = json.loads(cleaned)
                if isinstance(arr, list):
                    for item in arr:
                        w = item.get("word")
                        s = item.get("sentence", "").strip()
                        if w and s:
                            out[w] = s
            print(f"Batch sents parsed: {len(out)} items")
        except Exception as e:
            print(f"Batch sents error: {e} - per-word fallback")
    
    for w in words:
        if w not in out:
            try:
                p = f'Write one natural sentence (8-18 words) that would contain the word "{w}", but replace the word with "____". Return only the sentence, no extra text.'
                raw = invoke_llm(p)
                if raw:
                    cleaned = re.sub(r'```.*```', '', raw.strip(), flags=re.DOTALL).strip()
                    out[w] = cleaned.splitlines()[0]
                else:
                    out[w] = f"The ____ demonstrates how {w.lower()} is used in a sentence. (fallback)"
            except Exception as e:
                print(f"Per-word sent fail for {w}: {e}")
                out[w] = f"Fill in the ____ with the correct word. (error fallback)"
            # In generate_sentences, per-word loop (after except):
            if not out.get(w):
                # Word-specific fallback sentence
                if w.lower() in ['compose', 'defuse', 'antebellum']:
                    out[w] = {
                        'compose': 'The musician would ____ a beautiful melody on the piano every evening.',
                        'defuse': 'The negotiator tried to ____ the tense argument before it escalated.',
                        'antebellum': 'The ____ mansion stood as a reminder of the South\'s history before the war.'
                    }.get(w.lower(), f"The ____ illustrates a typical use of {w.lower()} in context. (fallback)")
                else:
                    out[w] = f"The ____ is where {w.lower()} fits naturally in a sentence. (fallback)"
    return out


def process_exercises(n: int, level: int, html: bool = False):
    redis = get_redis_connection()
    job_key = f"job:{time.time()}"
    redis.set(job_key, f"Processed {n} exercises at level {level}, html={html}")
    redis.expire(job_key, 3600)  # Expire after 1 hour
    conn = get_connection()
    try:
        rows = get_random_rows(conn, n, level)
        if not rows:
            return {"error": "No words found"}
        payload = build_exercises_from_rows(rows)
        # Ensure payload is UTF-8 compliant
        if isinstance(payload, str):
            payload = payload.encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(payload, (list, dict)):
            # Recursively encode strings in payload
            def encode_recursive(data):
                if isinstance(data, str):
                    return data.encode('utf-8', errors='replace').decode('utf-8')
                elif isinstance(data, list):
                    return [encode_recursive(item) for item in data]
                elif isinstance(data, dict):
                    return {k: encode_recursive(v) for k, v in data.items()}
                return data
            payload = encode_recursive(payload)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC (Custom Format)")
        if html:
            return {"html_data": {"payload": payload, "timestamp": timestamp, "level": level}}
        return {"exercises": payload, "timestamp": timestamp, "level": level}
    finally:
        conn.close()
