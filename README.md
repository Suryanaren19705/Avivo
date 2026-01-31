Mini RAG Telegram Bot

This project implements a lightweight Retrieval-Augmented Generation (RAG) chatbot using Telegram.
The bot retrieves relevant information from local documents using sentence embeddings and SQLite,
then generates responses using a small decoder-only language model.

This project demonstrates an end-to-end RAG pipeline:

User → Telegram Bot → Embeddings → SQLite Retrieval → Decoder LLM → Response


FEATURES

- Telegram chatbot interface
- Local document-based knowledge system
- Semantic search using MiniLM embeddings
- SQLite vector storage
- Decoder-only LLM for answer generation
- Multi-domain support (Finance, IT, Healthcare, Automobile)
- Deterministic generation with reduced hallucination


TECH STACK

- Python 3.10+
- python-telegram-bot
- sentence-transformers (all-MiniLM-L6-v2)
- SQLite
- HuggingFace Transformers
- Decoder-only LLM (gpt2)
- NumPy


PROJECT STRUCTURE

RAG_BOT/
|
├── bot.py
├── build_db.py
├── docs/
├── requirements.txt
├── rag.db
└── README.md


SETUP INSTRUCTIONS

Step 1 — Clone Repository

git clone https://github.com/Suryanaren19705/Avivo.git
cd Avivo


Step 2 — Create Environment

conda create -n rag_bot python=3.10
conda activate rag_bot


Step 3 — Install Requirements

pip install -r requirements.txt


Step 4 — Build SQLite Vector Database

python build_db.py

This creates rag.db.


Step 5 — Configure Telegram Bot Token

Open bot.py and replace:

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"


Step 6 — Run Bot

python bot.py

Terminal should show:

Bot running...


USAGE

Open Telegram and search for your bot username.

Ask questions using:

/ask <your question>

Example:

/ask What is loan default?


RAG FLOW

1. User sends query via Telegram
2. Query converted to embedding using MiniLM
3. SQLite searched with cosine similarity
4. Top relevant chunks retrieved
5. Retrieved context passed to decoder LLM
6. Final answer sent back to user


NOTES

- Decoder-only models are probabilistic and may hallucinate occasionally.
- Strict prompting and greedy decoding are used to reduce randomness.
- Recommended models for Windows CPU:
  - gpt2


AUTHOR

Surya Narayanan


LICENSE

Educational / Demonstration use.
