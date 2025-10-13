#!/usr/bin/env python3
"""
classify_levels_batch_chatollama.py

Batch-mode classifier for word_info_level.db -> word_info table.

Requirements:
- venv with langchain and langchain-community installed
- langchain_community.chat_models.ChatOllama available
- Ollama server running with model "mistral:7b" on localhost:11434

What it does:
- Reads all words from word_info table
- Sends them to ChatOllama in batches (BATCH_CLASSIFY_SIZE)
- Expects a JSON array response with objects:
  {"word":"...", "level":1..5, "confidence":0.00-1.00, "reason":"..."}
- Writes results back to DB in commits of DB_WRITE_BATCH_SIZE rows
- Falls back to per-word calls for failed batches
"""

import sqlite3
import json
import time
import re
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple

from langchain_community.chat_models import ChatOllama

# ----------------- CONFIG -----------------
DB_PATH = "word_info_level.db"
TABLE = "word_info"

# Batch size for LLM classification (words per LLM call). Start with 20, can try 50.
BATCH_CLASSIFY_SIZE = 20
# How many DB row-updates to buffer before writing to disk
DB_WRITE_BATCH_SIZE = 200

# LLM and retry settings
BATCH_RETRIES = 2
BATCH_RETRY_DELAY = 1.0  # seconds
SINGLE_RETRIES = 2
SINGLE_RETRY_DELAY = 0.6

# Progress printing
PRINT_EVERY_N_BATCHES = 1

# ChatOllama setup
CHAT_MODEL = "mistral:7b"
CHAT_BASE_URL = "http://localhost:11434"
# instantiate ChatOllama (temperature 0.0 recommended if supported)
chat = ChatOllama(model=CHAT_MODEL, base_url=CHAT_BASE_URL, temperature=0.0)

# ----------------- DB context -----------------
@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()

# ----------------- Utilities -----------------
def sanitize_word(w: Optional[str]) -> str:
    if w is None:
        return ""
    s = str(w).strip()
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    # Remove surrounding quotes if present
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def extract_json_array_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Robustly extract a JSON array from possible surrounding text.
    Returns list of dicts or None.
    """
    if not text:
        return None
    s = text.strip()
    # Remove ``` fenced blocks and keep inner content if present
    if s.startswith("```"):
        parts = s.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("[") or p2.startswith("{"):
                s = p2
                break
    # Find first '[' and last ']'
    first = s.find('[')
    last = s.rfind(']')
    candidate = None
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last+1]
    else:
        # maybe model returned just JSON without array but with objects concatenated
        candidate = s
    # Try parsing
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
        # if it's an object, wrap it
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        # fallback: attempt to find {...}{...} patterns (rare). Skip.
        return None
    return None

# ----------------- Prompt builders -----------------
def build_batch_prompt(words: List[str]) -> str:
    """
    Build a compact batch prompt asking for JSON array.
    """
    words_json = json.dumps(words, ensure_ascii=False)
    prompt = (
        "You are a strict JSON-only classifier. For each input word return an object:\n"
        '{"word":"...","level":1,"confidence":0.00,"reason":"short"}\n\n'
        "Levels (choose exactly one):\n"
        "- 1: Very basic everyday (family, food, travel)\n"
        "- 2: Common concrete (objects, hobbies, jobs)\n"
        "- 3: Society/institutions (news, law, school, workplace)\n"
        "- 4: Abstract/conceptual (ideas, emotions, argumentation)\n"
        "- 5: Technical/specialized (science, medicine, engineering)\n\n"
        "Return ONLY a JSON ARRAY in the SAME ORDER as the INPUT_WORDS. Example:\n"
        '[{"word":"dog","level":1,"confidence":0.95,"reason":"common noun"}]\n\n'
        "INPUT_WORDS:\n"
        + words_json
        + "\n\nReturn ONLY the JSON array now."
    )
    return prompt

def build_single_prompt(word: str) -> str:
    return (
        "You are a strict JSON-only classifier. Return a single JSON object with fields:\n"
        '{"word":"...","level":1,"confidence":0.00,"reason":"short"}\n\n'
        "Levels:\n"
        "- 1: Very basic everyday (family, food, travel)\n"
        "- 2: Common concrete (objects, hobbies, jobs)\n"
        "- 3: Society/institutions (news, law, school, workplace)\n"
        "- 4: Abstract/conceptual (ideas, emotions, argumentation)\n"
        "- 5: Technical/specialized (science, medicine, engineering)\n\n"
        f'Now classify this word: "{word}"\nReturn ONLY the JSON object.'
    )

# ----------------- LLM call helpers -----------------
def invoke_or_predict(prompt: str) -> str:
    """
    Try chat.invoke(prompt).content, else fallback to predict() return value coerced to string.
    """
    try:
        # prefer invoke (langchain deprecation fallback)
        resp = chat.invoke(prompt)
        # resp may be an object with 'content' attribute or may be a string
        if hasattr(resp, "content"):
            return resp.content
        # some wrappers return list or dict; stringify
        return str(resp)
    except Exception:
        # fallback to predict (older method)
        try:
            resp2 = chat.predict(prompt)
            return resp2 if isinstance(resp2, str) else str(resp2)
        except Exception as e:
            raise

def classify_batch_llm(words: List[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Return list of dicts with keys word, level, confidence, reason in same order as input,
    or None on failure.
    """
    prompt = build_batch_prompt(words)
    last_err = None
    for attempt in range(1, BATCH_RETRIES + 1):
        try:
            text = invoke_or_predict(prompt)
            arr = extract_json_array_from_text(text)
            if arr is None:
                raise ValueError("could not parse JSON array")
            # Normalize and validate
            out = []
            for obj in arr:
                if not isinstance(obj, dict):
                    out.append(None)
                    continue
                w = obj.get("word")
                lvl = obj.get("level")
                conf = obj.get("confidence", 0.6)
                reason = obj.get("reason", "") or ""
                # normalize
                try:
                    lvl = int(lvl)
                    if lvl < 1 or lvl > 5:
                        raise ValueError()
                except Exception:
                    lvl = 3
                try:
                    conf = float(conf)
                    conf = max(0.0, min(1.0, conf))
                except Exception:
                    conf = 0.6
                out.append({"word": w, "level": lvl, "confidence": conf, "reason": reason})
            # length must match
            if len(out) != len(words):
                raise ValueError(f"length mismatch: got {len(out)} expected {len(words)}")
            return out
        except Exception as e:
            last_err = repr(e)
            time.sleep(BATCH_RETRY_DELAY * attempt)
    # give up on batch
    return None

def classify_single_llm(word: str, max_attempts: int = SINGLE_RETRIES) -> Tuple[int, float, str]:
    word_clean = sanitize_word(word)
    if not word_clean:
        return 3, 0.2, "empty-or-invalid"
    prompt = build_single_prompt(word_clean)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            text = invoke_or_predict(prompt)
            # try parse object
            parsed = None
            # extract object even if wrapped
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    pass
                elif isinstance(parsed, list) and parsed:
                    parsed = parsed[0] if isinstance(parsed[0], dict) else None
                else:
                    parsed = None
            except Exception:
                # fallback to robust extraction of object
                first = text.find('{')
                last = text.rfind('}')
                if first != -1 and last != -1 and last > first:
                    try:
                        parsed = json.loads(text[first:last+1])
                    except Exception:
                        parsed = None
            if isinstance(parsed, dict):
                lvl = parsed.get("level")
                try:
                    lvl = int(lvl)
                    if lvl < 1 or lvl > 5:
                        raise ValueError()
                except Exception:
                    lvl = 3
                conf = parsed.get("confidence", 0.6)
                try:
                    conf = float(conf)
                    conf = max(0.0, min(1.0, conf))
                except Exception:
                    conf = 0.6
                reason = parsed.get("reason", "") or ""
                return lvl, conf, reason
            else:
                last_err = f"no-parse:{repr(text[:200])}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(SINGLE_RETRY_DELAY * attempt)
    return 3, 0.2, f"fallback:{last_err}"

# ----------------- DB helpers -----------------
def ensure_columns(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({TABLE})")
    cols = [r[1] for r in cur.fetchall()]
    altered = False
    if "level" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN level INTEGER")
        altered = True
    if "confidence" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN confidence REAL")
        altered = True
    if "reason" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN reason TEXT")
        altered = True
    if altered:
        conn.commit()

def fetch_all_words(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    cur = conn.cursor()
    cur.execute(f"SELECT rowid, word FROM {TABLE}")
    return cur.fetchall()

def update_batch(conn: sqlite3.Connection, updates: List[Tuple[int, str, float, int]]):
    """
    updates: list of tuples (level, reason, confidence, rowid)
    """
    cur = conn.cursor()
    cur.executemany(
        f"UPDATE {TABLE} SET level = ?, reason = ?, confidence = ? WHERE rowid = ?",
        updates
    )
    conn.commit()

# ----------------- Main -----------------
def main():
    print("Starting batch classification")
    start_all = time.time()
    with get_connection() as conn:
        ensure_columns(conn)
        rows = fetch_all_words(conn)
        total = len(rows)
        print(f"Found {total} words in {TABLE}")

        db_buffer: List[Tuple[int, str, float, int]] = []
        i = 0
        batch_count = 0
        while i < total:
            chunk = rows[i:i+BATCH_CLASSIFY_SIZE]
            ids = [r[0] for r in chunk]
            words = [sanitize_word(r[1]) for r in chunk]

            # call batch classifier
            t0 = time.time()
            result = classify_batch_llm(words)
            t1 = time.time()
            batch_count += 1

            if result is None:
                # fallback: per-word classification for this chunk
                print(f"[Batch {batch_count}] Batch failed, falling back to per-word for chunk starting at {i+1}")
                for rid, w in zip(ids, words):
                    lvl, conf, reason = classify_single_llm(w)
                    db_buffer.append((lvl, reason, conf, rid))
            else:
                # success: map result to db_buffer
                for item, rid in zip(result, ids):
                    if not item or not isinstance(item, dict):
                        lvl, conf, reason = 3, 0.2, "invalid-item"
                    else:
                        lvl = item.get("level", 3)
                        conf = item.get("confidence", 0.6)
                        reason = item.get("reason", "") or ""
                        # normalization
                        try:
                            lvl = int(lvl)
                            if lvl < 1 or lvl > 5:
                                lvl = 3
                        except Exception:
                            lvl = 3
                        try:
                            conf = float(conf)
                        except Exception:
                            conf = 0.6
                    db_buffer.append((lvl, reason, conf, rid))

            # print timing/progress
            elapsed = t1 - t0
            processed = min(i + BATCH_CLASSIFY_SIZE, total)
            print(f"[{processed}/{total}] batch {batch_count} processed in {elapsed:.2f}s (buffered {len(db_buffer)} rows)")

            # flush DB buffer periodically
            if len(db_buffer) >= DB_WRITE_BATCH_SIZE:
                t_flush = time.time()
                update_batch(conn, db_buffer)
                t_flush_elapsed = time.time() - t_flush
                print(f"  -> DB flushed {len(db_buffer)} rows in {t_flush_elapsed:.2f}s")
                db_buffer = []

            i += BATCH_CLASSIFY_SIZE

        # final flush
        if db_buffer:
            t_flush = time.time()
            update_batch(conn, db_buffer)
            print(f"Final DB flush {len(db_buffer)} rows in {time.time()-t_flush:.2f}s")

    total_elapsed = time.time() - start_all
    print(f"Done. Time elapsed: {total_elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()



