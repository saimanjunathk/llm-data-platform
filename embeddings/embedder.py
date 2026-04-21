# This file converts text into vector embeddings
# Embeddings are numerical representations of text
# Similar texts have similar vectors (close in vector space)
#
# Example:
# "What is machine learning?" → [0.23, -0.45, 0.12, ...]  (384 numbers)
# "What is deep learning?"    → [0.21, -0.43, 0.15, ...]  (similar!)
# "What is the weather?"      → [0.89,  0.12, -0.67, ...] (different!)
#
# We use sentence-transformers — free, runs locally, no API key needed
# Model: all-MiniLM-L6-v2 — small, fast, good quality

import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextEmbedder:

    # model_name → which sentence-transformer model to use
    # all-MiniLM-L6-v2 is the best balance of speed and quality
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        logger.info(f"Loading embedding model: {model_name}")

        # SentenceTransformer downloads model on first run
        # Cached locally after that
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension  = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

        logger.info(f"Model loaded! Embedding dimension: {self.dimension}")


    # This METHOD converts a single text to a vector
    def embed_text(self, text: str) -> np.ndarray:

        # encode() converts text to numpy array
        # normalize_embeddings=True → vectors have length 1
        # Normalized vectors make cosine similarity = dot product (faster)
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding


    # This METHOD converts multiple texts to vectors at once
    # Much faster than calling embed_text() in a loop
    def embed_batch(self, texts: list) -> np.ndarray:

        logger.info(f"Embedding {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,       # process 32 texts at a time
            show_progress_bar=False
        )

        logger.info(f"Generated {len(embeddings)} embeddings of dim {self.dimension}")
        return embeddings


    # This METHOD chunks long text into smaller pieces
    # LLMs have context limits — we can't embed 100-page documents at once
    # We split into overlapping chunks so context isn't lost at boundaries
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list:

        # Split text into words
        words = text.split()

        chunks = []
        start  = 0

        while start < len(words):
            # Take chunk_size words
            end   = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move forward by chunk_size - overlap
            # overlap means consecutive chunks share some words
            # This ensures context isn't lost at chunk boundaries
            start += chunk_size - overlap

        return chunks


if __name__ == "__main__":
    embedder = TextEmbedder()

    texts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is the weather today?",
        "How does a neural network work?"
    ]

    embeddings = embedder.embed_batch(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Calculate similarity between first two texts
    # Dot product of normalized vectors = cosine similarity
    sim_12 = np.dot(embeddings[0], embeddings[1])
    sim_13 = np.dot(embeddings[0], embeddings[2])

    print(f"\nSimilarity: '{texts[0]}' vs '{texts[1]}': {sim_12:.4f}")
    print(f"Similarity: '{texts[0]}' vs '{texts[2]}': {sim_13:.4f}")
    print("(Higher = more similar)")