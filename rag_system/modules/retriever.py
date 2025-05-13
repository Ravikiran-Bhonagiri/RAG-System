# rag_system/modules/retriever.py
from typing import List, Dict
from langchain_core.retrievers import BaseRetriever
from .vector_db import VectorDB #Importing the vectorDB class
from langchain.schema import Document #Importing Document class

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