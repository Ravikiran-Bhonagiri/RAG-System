# rag_system/modules/embedder.py
from typing import List, Literal, Optional
import os  # For accessing environment variables
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

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
    

class Embedder_V2:
    """
    Handles text embeddings with configurable models: Hugging Face, Ollama, and OpenAI.
    It securely retrieves the OpenAI API key from the environment.
    It retrieves the Ollama base URL from the environment.
    """

    def __init__(
        self,
        embedding_model: Literal["openai", "huggingface", "ollama"] = "openai",
        model_name: str = "sentence-transformers/all-mpnet-base-v2",  # Generic model name
    ):
        """
        Initializes the Embedder.
        Args:
            embedding_model: Which embedding model type to use:"openai", "huggingface", or "ollama"
            model_name: The specific model to use. For OpenAI, this is the embedding model name (e.g., "text-embedding-ada-002").
                         For HuggingFace, it's the model identifier (e.g., "sentence-transformers/all-mpnet-base-v2").
                         For Ollama, it's the model name (e.g, "llama2").
        """
        self.embedding_model = embedding_model.lower()
        self.model_name = model_name
        self.embedder = self._initialize_embedder()


    def _initialize_embedder(self) -> Embeddings:
        """Initializes the chosen embedding model."""
        if self.embedding_model == "openai":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using the OpenAI embedding model."
                )
            return OpenAIEmbeddings(openai_api_key=openai_api_key, model=self.model_name)

        elif self.embedding_model == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name)

        elif self.embedding_model == "ollama":
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL") # retrieve from env
            ollama_kwargs = {}
            if ollama_base_url:
                ollama_kwargs["base_url"] = ollama_base_url
            return OllamaEmbeddings(model=self.model_name, **ollama_kwargs)

        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        return self.embedder.embed_documents(documents)

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embedder.embed_query(query)