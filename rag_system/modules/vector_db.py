# rag_system/modules/vector_db.py
from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
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


class AbstractVectorDB(ABC):
    """
    Abstract base class for managing vector database operations.
    This defines the interface that concrete VectorDB classes must implement.
    """

    @abstractmethod
    def __init__(self, embedder: Embedder):
        """
        Initializes the VectorDB with an Embedder instance.

        Args:
            embedder: The Embedder instance to use for generating embeddings.  Must be an instance of Embedder class
        """
        self.embedder = embedder
        self.vectorstore: VectorStore | None = None

    @abstractmethod
    def create_index(self, chunks: List[Dict]) -> None:
        """
        Creates a vector index from document chunks.

        Args:
            chunks: A list of dictionaries, where each dictionary represents a document chunk
                    and contains 'page_content' and 'metadata' keys.
        """
        pass

    @abstractmethod
    def save_index(self, path: str) -> None:
        """
        Saves the vector index to disk.

        Args:
            path: The path to save the vector index to.
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        """
        Loads a vector index from disk.

        Args:
            path: The path to load the vector index from.
        """
        pass


class FAISSVectorDB(AbstractVectorDB):
    """Manages vector database operations using FAISS."""

    def __init__(self, embedder: Embedder):
        super().__init__(embedder)
        self.embedder = embedder  # Type hint added
        self.vectorstore: FAISS | None = None # Narrow the type hint

    def create_index(self, chunks: List[Dict]) -> None:
        """Create a FAISS vector index from document chunks"""
        langchain_docs = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
        self.vectorstore = FAISS.from_documents(langchain_docs, self.embedder.embedder)

    def save_index(self, path: str) -> None:
        """Save the FAISS vector index to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_index(self, path: str) -> None:
        """Load a FAISS vector index from disk"""
        self.vectorstore = FAISS.load_local(path, self.embedder.embedder, allow_dangerous_deserialization=True)


class ChromaVectorDB(AbstractVectorDB):
    """Manages vector database operations using Chroma."""

    def __init__(self, embedder: Embedder, persist_directory: str = "chroma_db"):
        super().__init__(embedder)
        self.embedder = embedder
        self.persist_directory = persist_directory
        self.vectorstore: Chroma | None = None

    def create_index(self, chunks: List[Dict]) -> None:
        """Create a Chroma vector index from document chunks"""
        langchain_docs = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
        self.vectorstore = Chroma.from_documents(langchain_docs, self.embedder.embedder, persist_directory=self.persist_directory)
        self.vectorstore.persist()  # Ensure data is written to disk

    def save_index(self, path: str) -> None:
        """
        Chroma's save is handled by its persist_directory; this method does nothing.
        """
        print("Chroma's save operation is handled by its persist_directory.  This method does nothing.") # Alert the user
        pass

    def load_index(self, path: str) -> None:
         """Load a Chroma vector index from disk (actually, from the persist directory)"""
         self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder.embedder)