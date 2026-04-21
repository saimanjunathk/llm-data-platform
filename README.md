# LLM Data Platform

## Live Demo
View Live Dashboard: https://llm-data-platform.streamlit.app/

## Architecture
Documents -> Chunking -> Embeddings (Sentence Transformers) -> ChromaDB Vector Store -> RAG Pipeline -> Claude LLM -> Streamlit Chat Interface

## Features
- RAG Chat: Ask questions about company reports and financial documents
- AI SQL Agent: Natural language to SQL queries powered by Claude
- Vector Store Explorer: Search and explore the knowledge base

## Tech Stack
- LLM: Anthropic Claude (claude-haiku-4-5)
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- RAG: Custom retrieval pipeline
- SQL Agent: Claude-powered NL to SQL
- Dashboard: Streamlit

## How to Run Locally
git clone https://github.com/saimanjunathk/llm-data-platform
cd llm-data-platform
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Add ANTHROPIC_API_KEY=your-key to .env file
streamlit run dashboard/app.py

## Status
- Document Ingestion - Done
- Vector Embeddings - Done
- ChromaDB Vector Store - Done
- RAG Pipeline - Done
- AI SQL Agent - Done
- Live Dashboard - Done