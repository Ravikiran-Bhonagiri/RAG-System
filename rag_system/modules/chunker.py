# rag_system/modules/chunker.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class Chunker:
    """Handles document chunking with configurable settings"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""

        langchain_docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
        chunks = self.text_splitter.split_documents(langchain_docs)
        return [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]