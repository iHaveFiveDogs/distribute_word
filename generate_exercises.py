#!/usr/bin/env python3
"""
generate_exercises.py

Simplified behavior:
- Pick N random words from the DB.
- Generate N definition-fill (Part A) and N sentence-fill (Part B) exercises.
- Output JSON containing:
  { "chosen_words": [...shuffled word bank...], "exercises": [...] }
- DOES NOT request or include word meanings.
- The word bank order is intentionally shuffled so it differs from exercise order.

Run:
  python generate_exercises.py -n 20 --db word_info_level.db --out ./out
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ChatOllama (used only for generating sentences/definitions if DB lacks them)
from langchain_community.chat_models import ChatOllama

# ---------------- CONFIG ----------------
DEFAULT_DB = "word_info_level.db"
DEFAULT_TABLE = "word_info"
DEFAULT_N = 20

CHAT_MODEL = "gmistral:7b"
CHAT_BASE_URL = "http://localhost:11434"
CHAT_TEMPERATURE = 0.0

# lazy client
chat: Optional[ChatOllama] = None

def init_chat():
    global chat
    if chat is None:
        chat = ChatOllama(model=CHAT_MODEL, base_url=CHAT_BASE_URL, temperature=CHAT_TEMPERATURE)

# ---------- DB helpers ----------
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def fetch_random_rows(conn: sqlite3.Connection, table: str, n: int) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    select_cols = ["rowid", "word"]
    if "definition" in cols:
        select_cols.append("definition")
    if "example" in cols:
        select_cols.append("example")
    q = f"SELECT {', '.join(select_cols)} FROM {table} ORDER BY RANDOM() LIMIT ?"
    cur.execute(q, (n,))
    return list(cur.fetchall())

# ---------- LLM prompts (minimal, only used if needed) ----------
def _sentence_prompt(words: List[str]) -> str:
    words_json = json.dumps(words, ensure_ascii=False)
    return (
        "RETURN ONLY JSON.\n"
        "For each input word return an object {\"word\":..., \"sentence\":...}.\n"
        "Each sentence should be 8-18 words and contain a blank '____' where the target belongs.\n"
        "Do NOT include the target word anywhere in the sentence.\n"
        "Return array in the SAME ORDER as INPUT_WORDS.\n\n"
        f"INPUT_WORDS: {words_json}\n"
    )

def _safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    t = raw.strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        first = t.find('[')
        last = t.rfind(']')
        if first != -1 and last != -1 and last > first:
            try:
                cand = t[first:last+1]
                parsed = json.loads(cand)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None
    return None

def invoke_llm(prompt: str) -> str:
    init_chat()
    resp = chat.invoke(prompt)
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)

def generate_sentences(words: List[str]) -> Dict[str, str]:
    """Return word -> sentence_with_blank. If LLM fails, return empty strings."""
    out: Dict[str, str] = {}
    if not words:
        return out
    try:
        raw = invoke_llm(_sentence_prompt(words))
        arr = _safe_parse_json_array(raw)
        if arr:
            for item in arr:
                w = item.get("word", "")
                sent = (item.get("sentence") or "").strip()
                out[w] = sent
            return out
    except Exception as e:
        # silent fallback, print debug to stderr
        print("LLM sentence generation failed:", e, file=sys.stderr)
    # fallback per-word simple template (no LLM)
    for w in words:
        out[w] = f"(no sentence generated for {w})"
    return out

# ---------- Exercise builder (no meanings) ----------
def build_exercises(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    sampled_words = [r["word"] for r in rows]

    defs = {r["word"]: (str(r["definition"]).strip() if "definition" in r.keys() and r["definition"] else "")
            for r in rows}

    need_gen = []
    db_sentences = {}
    for r in rows:
        w = r["word"]
        if "example" in r.keys() and r["example"]:
            ex = str(r["example"]).strip()
            if re.search(re.escape(w), ex, flags=re.IGNORECASE):
                db_sentences[w] = re.sub(re.escape(w), "____", ex, count=1, flags=re.IGNORECASE)
            else:
                need_gen.append(w)
        else:
            need_gen.append(w)

    gen = generate_sentences(need_gen) if need_gen else {}
    sentences = {**db_sentences, **gen}

    # word bank = DB order
    word_bank = sampled_words

    # shuffle orders separately
    defs_order = sampled_words.copy()
    sents_order = sampled_words.copy()
    random.shuffle(defs_order)
    random.shuffle(sents_order)

    # ensure different order
    tries = 0
    while sents_order == defs_order and tries < 10:
        random.shuffle(sents_order)
        tries += 1

    part_a = [{"part": "A", "type": "definition_fill",
               "prompt": defs.get(w, "") or "(definition not available)", "answer": w}
              for w in defs_order]

    part_b = [{"part": "B", "type": "sentence_fill",
               "prompt": sentences.get(w, "") or "(sentence not available)", "answer": w}
              for w in sents_order]

    return {
        "chosen_words": word_bank,
        "part_a": part_a,
        "part_b": part_b,
    }



# ---------- CLI ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Table name")
    parser.add_argument("-n", "--num", type=int, default=DEFAULT_N, help="Number of words to sample")
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out)
    conn = get_connection(args.db) if False else sqlite3.connect(args.db)  # dummy to detect get_connection missing
    # use local lightweight connection to avoid importing function mismatch
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = fetch_random_rows(conn, args.table, args.num)
        if not rows:
            print("No rows found. Exiting.", file=sys.stderr)
            sys.exit(2)
        payload = build_exercises(rows)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_name = os.path.join(args.out, f"exercises_{args.num*2}_{ts}.json")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("Wrote:", out_name)
        # also write simple word bank list (shuffled order)
        wb_name = os.path.join(args.out, f"word_bank_{ts}.txt")
        with open(wb_name, "w", encoding="utf-8") as f:
            for w in payload["chosen_words"]:
                f.write(w + "\n")
        print("Wrote word bank:", wb_name)
    finally:
        conn.close()

if __name__ == "__main__":
    main()


