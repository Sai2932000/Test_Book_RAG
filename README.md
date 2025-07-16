# 📚 Textbook Q&A with RAG using Vertex AI + LangChain

This project is a **Retrieval-Augmented Generation (RAG)** based Q&A system that allows users to ask questions directly from a textbook or PDF and receive contextual, accurate answers using **LLMs**.

Built with **LangChain**, **ChromaDB**, and **Google Vertex AI**, this project demonstrates how GenAI can be used to make studying more interactive and efficient.

---

## 🧠 What It Does

- Takes a textbook or long PDF as input
- Splits content into chunks and embeds it using **Vertex AI Embeddings**
- Stores chunks in a **ChromaDB vector store**
- On user question, retrieves the most relevant chunks
- Uses **Vertex AI chat model** to generate a contextual answer based on the retrieved content

---

## 🚀 Tech Stack

- 🐍 Python 3
- 🦜 LangChain
- 📦 ChromaDB (for vector search)
- 🔍 Google Vertex AI (for embeddings + LLM)
- 🧠 NLP Techniques (chunking, semantic search)
- Optional UI: Streamlit (if added)

---

## 📦 Installation

```bash
pip install -r requirements.txt
