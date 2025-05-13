# rag_system/system/rag_system.py
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

# Import modules, using the new 'core' and 'ingestion' directory names
from rag_system.modules import Chunker, Embedder, VectorDB, Retriever, QAChain, RAGPrompt, WebDocumentLoader, PDFDocumentLoader

class RAGSystem:
    """Main RAG system class that orchestrates all components"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.document_loader = None
        self.chunker = None
        self.embedder = None
        self.vector_db = None
        self.retriever = None
        self.qa_chain = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all RAG components based on config"""
        # Initialize with default values or from config
        chunk_size = self.config.get("chunk_size", 1000)
        chunk_overlap = self.config.get("chunk_overlap", 200)
        embedding_model = self.config.get("embedding_model", "huggingface")
        search_type = self.config.get("search_type", "similarity")
        retrieval_k = self.config.get("retrieval_k", 4)
        llm_model = self.config.get("llm_model", "gpt-3.5-turbo") #Get all the config parameters and assign them to the global self variable
        rag_template = self.config.get('prompt_template') #Getting the rag template

        self.chunker = Chunker(chunk_size, chunk_overlap) #Creating class objects using the config parameters
        self.embedder = Embedder(embedding_model)
        self.vector_db = VectorDB(self.embedder)
        # self.retriever = Retriever(self.vector_db, search_type, retrieval_k) # REMOVE THIS LINE- Initialized after creating index

        # Initialize LLM
        llm = ChatOpenAI(model=llm_model, temperature=0) #Creating the llm with ChatOpenAI class and passing the model through configuration

        # Initialize QA Chain
        rag_prompt = RAGPrompt(rag_template) #Initiliazing the rag prompt class

        # self.qa_chain = QAChain(self.retriever, llm, rag_prompt) # REMOVE THIS LINE - Initialized after creating index

    def load_documents(self, source: str, source_type: str = "web") -> None: #Here you are passing the type of document so you will need to import the documentloaders
        """Load documents from a source"""
        if source_type == "web":
            self.document_loader = WebDocumentLoader()
        elif source_type == "pdf":
            self.document_loader = PDFDocumentLoader()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        self.documents = self.document_loader.load(source)

    def process_documents(self) -> None:
        """Process loaded documents through the RAG pipeline"""
        if not hasattr(self, 'documents'):
            raise ValueError("No documents loaded. Call load_documents() first.")

        # Chunk documents
        chunks = self.chunker.chunk_documents(self.documents)

        # Create vector index
        self.vector_db.create_index(chunks)

        # AFTER the VectorDB is initialized...
        self.retriever = Retriever(self.vector_db, self.config.get("search_type", "similarity"),
                                    self.config.get("retrieval_k", 4))  # Initialzing retriever after the vector dB has been initalized

        # Re intialize again...
        llm = ChatOpenAI(model=self.config.get("llm_model", "gpt-3.5-turbo"),
                         temperature=0)  # Creating the base LLM to pass to the QAChain object

        rag_prompt = RAGPrompt(self.config.get("prompt_template"))  # Creating baseragPrompt object

        self.qa_chain = QAChain(self.retriever, llm, rag_prompt)  # Finally intializing the QAchain object

    def query(self, question: str) -> str:
        """Query the RAG system with a question"""
        if not self.qa_chain:
            raise ValueError("RAG system not fully initialized. Call process_documents() first.")
        return self.qa_chain.ask(question)

    def save_system(self, path: str) -> None:
        """Save the RAG system state to disk"""
        if not self.vector_db:
            raise ValueError("Vector database not initialized")
        self.vector_db.save_index(path)

    def load_system(self, path: str) -> None:
        """Load a RAG system state from disk"""
        if not self.vector_db:
            self._initialize_components()
        self.vector_db.load_index(path)
