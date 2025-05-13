# main.py
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.append("D:/RAG")

from rag_system.system import RAGSystem
# Example usage
if __name__ == "__main__":
    # Initialize RAG system with configuration
    config = {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "embedding_model": "huggingface",
        "search_type": "similarity",
        "retrieval_k": 5,
        "llm_model": "gpt-3.5-turbo"  # Add this if not already present
    }

    rag = RAGSystem(config)

    # Load documents (web page or PDF)
    try:
        rag.load_documents("D:/RAG/fundamentals/documents/paper.pdf", "pdf")
        # rag.load_documents("path/to/document.pdf", "pdf")

        # Process documents
        rag.process_documents()

        # Query the system
        question = "What are large language models, what is speciality about phi4 model?"
        answer = rag.query(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        # Save the system for later use
        rag.save_system("rag_index")

    except Exception as e:
        print(f"An error occurred: {str(e)}")