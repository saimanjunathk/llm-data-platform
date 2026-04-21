# RAG = Retrieval Augmented Generation
# Instead of relying only on LLM's training knowledge,
# we RETRIEVE relevant documents and give them to the LLM as context
#
# Without RAG:
# User: "What did Company X report in their annual report?"
# LLM:  "I don't know" (not in training data)
#
# With RAG:
# 1. Search vector DB for Company X documents
# 2. Find relevant chunks from annual report
# 3. Give chunks to LLM as context
# 4. LLM answers based on actual document content!

import anthropic
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self, embedder, vector_store):
        self.embedder     = embedder
        self.vector_store = vector_store

        # Initialize Anthropic client
        # Reads ANTHROPIC_API_KEY from .env file automatically
        try:
            import streamlit as st
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            api_key = os.getenev("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5"
        # claude-3-haiku → fastest and cheapest Claude model
        # Perfect for RAG where we make many API calls

        logger.info("RAG pipeline initialized")


    # This METHOD retrieves relevant documents for a query
    def retrieve(self, query: str, n_results: int = 5) -> list:

        # Convert query to embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector store for similar documents
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results
        )

        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results


    # This METHOD generates an answer using retrieved context
    def generate(self, query: str, context_docs: list) -> dict:

        # Build context string from retrieved documents
        context = "\n\n---\n\n".join([
            f"Document: {doc['metadata'].get('title', doc['id'])}\n{doc['content']}"
            for doc in context_docs
        ])

        # Build prompt with context
        # This is the core of RAG: inject retrieved context into prompt
        prompt = f"""You are a helpful financial analyst assistant.
Answer the question based on the provided context documents.
If the answer is not in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

Please provide a clear, concise answer based on the context above."""

        logger.info("Calling Claude API...")

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.content[0].text

        return {
            "query":    query,
            "answer":   answer,
            "sources":  [doc["id"] for doc in context_docs],
            "context":  context_docs
        }


    # This METHOD runs the full RAG pipeline: retrieve + generate
    def ask(self, query: str, n_results: int = 5) -> dict:

        # Step 1: Retrieve relevant documents
        context_docs = self.retrieve(query, n_results)

        # Step 2: Generate answer with context
        result = self.generate(query, context_docs)

        return result