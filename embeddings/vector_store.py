# This file manages our ChromaDB vector database
# ChromaDB stores text chunks + their embeddings
# Allows us to search for similar chunks using vector similarity
#
# How vector search works:
# 1. Query: "What is momentum investing?"
# 2. Convert query to vector: [0.23, -0.45, 0.12, ...]
# 3. Find stored vectors closest to query vector
# 4. Return the text chunks those vectors came from
# 5. Feed those chunks to LLM as context

import chromadb
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStore:

    # persist_dir   → where ChromaDB saves data to disk
    # collection_name → name of our document collection
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "documents"
    ):
        os.makedirs(persist_dir, exist_ok=True)

        # chromadb.PersistentClient saves data to disk
        # So we don't re-embed documents every time we restart
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create collection
        # A collection is like a table in a regular database
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine distance = 1 - cosine similarity
            # Better than euclidean for text embeddings
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Vector store ready | Collection: {collection_name} | "
                    f"Documents: {self.collection.count()}")


    # This METHOD adds documents to the vector store
    # documents → list of dicts with id, content, metadata
    # embeddings → numpy array of embeddings for each document
    def add_documents(self, documents: list, embeddings: np.ndarray):

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")

        # ChromaDB expects lists not numpy arrays
        ids        = [doc["id"] for doc in documents]
        contents   = [doc["content"] for doc in documents]
        metadatas  = [doc.get("metadata", {}) for doc in documents]

        # Convert metadata values to strings (ChromaDB requirement)
        metadatas = [
            {k: str(v) for k, v in m.items()}
            for m in metadatas
        ]

        # add() stores documents with their embeddings
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents to vector store")


    # This METHOD searches for similar documents
    # query_embedding → embedding of the search query
    # n_results       → how many similar docs to return
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> list:

        # query() finds most similar vectors
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count())
        )

        # Format results into clean list of dicts
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "id":       results["ids"][0][i],
                "content":  results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
                # distance: lower = more similar (cosine distance)
            })

        return docs


    # This METHOD returns total document count
    def count(self) -> int:
        return self.collection.count()


    # This METHOD clears all documents
    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store cleared")