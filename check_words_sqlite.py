# check_first10_words_sqlite.py
import sqlite3
from contextlib import contextmanager

DB_PATH = "word_info_level.db"

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def ensure_columns(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(word_info)")
    cols = [r[1] for r in cur.fetchall()]
    print("Columns in 'word_info':", cols)
    # check expected columns
    for col in ["level", "confidence", "reason"]:
        if col not in cols:
            print(f"Column '{col}' is missing!")
        else:
            print(f"Column '{col}' exists.")

def count_all(conn):
    cur = conn.cursor()
    cur.execute("SELECT level, COUNT(*) FROM word_info GROUP BY level")
    rows = cur.fetchall()
    for row in rows:
        print(row)

def main():
    with get_connection() as conn:
        ensure_columns(conn)
        print("\nFirst 30 rows:")
        count_all(conn)

if __name__ == "__main__":
    main()
