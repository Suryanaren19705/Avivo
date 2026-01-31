from telegram.ext import ApplicationBuilder, CommandHandler
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

llm = pipeline("text-generation", model="gpt2")


BOT_TOKEN = "8020018548:AAGiBk4brdHW8E_T1ch5AoW-ezK7K_WJibE"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


conn = sqlite3.connect("rag.db", check_same_thread=False)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=2):
    q_emb = embed_model.encode(query)

    rows = conn.execute("SELECT content, embedding FROM docs").fetchall()

    scored = []
    for text, emb in rows:
        emb = np.frombuffer(emb, dtype=np.float32)
        scored.append((cosine(q_emb, emb), text))

    scored.sort(reverse=True)

    return [x[1] for x in scored[:top_k]]

async def ask(update, context):
    q = " ".join(context.args)

    if not q:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    docs = retrieve(q)

    context_text = "\n".join(docs)
    print(context_text)
    prompt = f"""
You are a question answering system.

RULES:
- Answer ONLY using the CONTEXT below.
- Do NOT add any new information.
- If the answer is not present, reply exactly: NOT FOUND.

CONTEXT:
{context_text}

QUESTION:
{q}

ANSWER:
"""


    result = llm(prompt,
                 do_sample=False,
                 max_length=500,repetition_penalty=1.1)[0]["generated_text"]

    answer = result.replace(prompt, "").strip()

    if not answer:
        answer = "Sorry, I could not generate a clear answer from the context."

    await update.message.reply_text(answer)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("ask", ask))

print("Bot running...")
app.run_polling()
