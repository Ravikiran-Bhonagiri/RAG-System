# rag_system/modules/vector_db.py
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings # Added import
from langchain.schema import Document

from .embedder import Embedder

class VectorDB:
    """Manages vector database operations"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.vectorstore = None  # Initialize vectorstore attribute

    def create_index(self, chunks: List[Dict]) -> None:
        """Create a vector index from document chunks"""

        langchain_docs = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
        self.vectorstore = FAISS.from_documents(langchain_docs, self.embedder.embedder)

    def save_index(self, path: str) -> None:
        """Save the vector index to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_index(self, path: str) -> None:
        """Load a vector index from disk"""
        self.vectorstore = FAISS.load_local(path, self.embedder.embedder, allow_dangerous_deserialization=True)