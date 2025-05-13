# rag_system/document_loaders/document_loader.py
from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, CSVLoader


class DocumentLoader(ABC):
    """Abstract base class for document loaders"""

    @abstractmethod
    def load(self, source: str) -> List[Dict]:
        """Load documents from a source"""
        pass


class PDFDocumentLoader(DocumentLoader):
    """Concrete implementation for loading PDF documents"""

    def load(self, file_path: str) -> List[Dict]:
        """Load documents from a PDF file"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    

class WebDocumentLoader(DocumentLoader):
    """Concrete implementation for loading web documents"""

    def load(self, url: str) -> List[Dict]:
        """Load documents from a web URL"""
        loader = WebBaseLoader(url)
        documents = loader.load()
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]


class TextDocumentLoader(DocumentLoader):
    """Concrete implementation for loading plain text documents"""

    def load(self, file_path: str) -> List[Dict]:
        """Load documents from a text file"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    

class CSVDocumentLoader(DocumentLoader):
    """Concrete implementation for loading CSV documents"""

    def load(self, file_path: str) -> List[Dict]:
        """Load documents from a CSV file"""
        loader = CSVLoader(file_path)
        documents = loader.load()
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]