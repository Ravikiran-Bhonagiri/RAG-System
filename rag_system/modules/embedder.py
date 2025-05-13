# rag_system/modules/embedder.py
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    """Handles text embeddings with configurable model"""

    def __init__(self, embedding_model: str = "openai"):
        self.embedding_model = embedding_model.lower()
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self) -> Embeddings:
        """Initialize the appropriate embedding model"""
        if self.embedding_model == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_model == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        return self.embedder.embed_documents(documents)

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embedder.embed_query(query)