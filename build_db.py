import sqlite3, os
import numpy as np
from sentence_transformers import SentenceTransformer
def build_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(text, size=200):
        return [text[i:i+size] for i in range(0, len(text), size)]

    conn = sqlite3.connect("rag.db")

    conn.execute("DROP TABLE IF EXISTS docs")

    conn.execute("""
    CREATE TABLE docs(
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding BLOB
    )
    """)

    for f in os.listdir("docs"):
        text = open(f"docs/{f}").read()

        chunks = chunk_text(text)

        for chunk in chunks:
            emb = model.encode(chunk)
            conn.execute(
                "INSERT INTO docs(content, embedding) VALUES (?,?)",
                (chunk, emb.tobytes())
            )

    conn.commit()
    print("SQLite DB rebuilt with chunks")
if __name__ == "__main__":
    build_db()