# rag_system/modules/chunker.py
from typing import List, Dict, Literal, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    MarkdownTextSplitter
)
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
    

class Chunker_V2:
    """Handles document chunking with configurable settings"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_method: Literal[
            "recursive", "character", "token", "nltk", "spacy", "markdown"
        ] = "recursive",
        spacy_model_name: str = "en_core_web_sm",  # Default Spacy model
        encoding_name = "gpt2", # Default encoder if required
        separator: str = "\n\n" # Seperators between the chars
    ):
        """
        Initializes the Chunker with configurable chunking settings.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks to maintain context.
            chunking_method: Which chunking method to use.
            spacy_model_name: The SpaCy model to load (e.g., "en_core_web_sm").
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        self.spacy_model_name = spacy_model_name
        self.encoding_name = encoding_name
        self.separator = separator
        self.text_splitter = self._create_text_splitter()

    def _create_text_splitter(self):
        """Creates the appropriate Langchain text splitter based on chunking_method."""
        if self.chunking_method == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        elif self.chunking_method == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator
            )
        elif self.chunking_method == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, encoding_name=self.encoding_name
            )
        elif self.chunking_method == "nltk":
            return NLTKTextSplitter(chunk_size=self.chunk_size)  # NLTK doesn't support chunk_overlap
        elif self.chunking_method == "spacy":
            return SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                pipeline=self.spacy_model_name,
            )
        elif self.chunking_method == "markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(
                f"Invalid chunking_method: {self.chunking_method}. Must be 'recursive', 'character', 'token', 'nltk', 'spacy', or 'markdown'."
            )

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""

        langchain_docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in documents
        ]
        chunks = self.text_splitter.split_documents(langchain_docs)
        return [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
