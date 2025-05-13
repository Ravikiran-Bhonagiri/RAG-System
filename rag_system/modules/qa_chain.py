# rag_system/modules/qa_chain.py
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .retriever import Retriever
from .prompt import RAGPrompt

class QAChain:
    """Handles question answering with RAG"""

    def __init__(self, retriever: Retriever, llm: BaseLanguageModel, prompt: RAGPrompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

        # Create the RAG chain
        self.chain = (
            {"context": retriever.retrieve_documents, "question": RunnablePassthrough()}  # Use retriever.retrieve_documents
            | self.prompt.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Ask a question and get an answer using the RAG chain"""
        return self.chain.invoke(question)