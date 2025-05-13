# rag_system/modules/retriever.py
from typing import List, Dict, Literal
from langchain_core.retrievers import BaseRetriever
from .vector_db import VectorDB #Importing the vectorDB class
from langchain.schema import Document #Importing Document class
from .vector_db import AbstractVectorDB

class Retriever:
    """Handles document retrieval with configurable strategies"""

    def __init__(self, vector_db: VectorDB, search_type: str = "similarity", k: int = 4):
        self.vector_db = vector_db
        self.search_type = search_type
        self.k = k
        self.base_retriever = None  # Initialize here - REMOVE THIS LINE
        # self.base_retriever = self._initialize_retriever()

    def _initialize_retriever(self) -> BaseRetriever:
        """Initialize the retriever with configured settings"""
        if self.vector_db.vectorstore is None:
            raise ValueError("Vector database not initialized. Create or load an index first.")
        return self.vector_db.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k}
        )

    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if self.base_retriever is None:  # ENSURE IT'S INITIALIZED LATE
            self.base_retriever = self._initialize_retriever()

        docs = self.base_retriever.invoke(query)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    


class Retriever_V2:
    """Handles document retrieval with configurable strategies, now backed by an ABC"""

    def __init__(
        self,
        vector_db: AbstractVectorDB,  # Use the ABC
        search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity", # Check the searches
        k: int = 4,
        fetch_k: int = 20, # Added Parameter
        score_threshold: float | None = None  # ADDED PARAMETER
    ):
        """
        Args:
            vector_db: Your initialized VectorDB object from the vector_db module inheriting from the ABC
            search_type: "similarity" to use similarity search or "mmr" to use Maximal Marginal Relevance search
            k: int = 4, # Number of documents to return.
            fetch_k: Number of documents to pass to MMR algorithm.
            score_threshold: Minimum similarity score to return results (for "similarity_score_threshold" search type)

        Raises:
            ValueError: If the vector database is not initialized.
        """
        self.vector_db = vector_db
        self.search_type = search_type
        self.k = k
        self.fetch_k = fetch_k # Added
        self.score_threshold = score_threshold
        self.base_retriever: BaseRetriever | None = None  # Initialize to None
        self._initialize_retriever()  # Eagerly initialize upon object creation

    def _initialize_retriever(self) -> None:  # No longer returns retriever
        """Initialize the retriever with configured settings"""
        if not self.vector_db.vectorstore:
            raise ValueError("Vector database not initialized.  Create or load an index first.")

        # Handle different search types
        if self.search_type == "similarity":
            search_kwargs = {"k": self.k} # Only calls K to ensure type
        elif self.search_type == "mmr":
             search_kwargs = {"k": self.k, "fetch_k": self.fetch_k} #Fetch K is the correct format.
        elif self.search_type == "similarity_score_threshold":  # New search type option

            if self.score_threshold is None:
                raise ValueError("score_threshold must be set for similarity_score_threshold search")
            search_kwargs = {"k": self.k, "score_threshold": self.score_threshold}

        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")

        self.base_retriever = self.vector_db.vectorstore.as_retriever(
            search_type=self.search_type, search_kwargs=search_kwargs
        )

    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.base_retriever:  # Double-check initialization (shouldn't be needed)
            self._initialize_retriever()  # Initialize if somehow not initialized
        docs: List[Document] = self.base_retriever.invoke(query)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
