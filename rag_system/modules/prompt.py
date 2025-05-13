# rag_system/modules/prompt.py
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

class RAGPrompt:
    """Manages RAG prompt templates and formatting"""

    def __init__(self, template: Optional[str] = None):
        self.template = template or self._get_default_template()
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _get_default_template(self) -> str:
        """Return a default RAG prompt template"""
        return """You are a helpful AI assistant. Use the following context to answer the question at the end.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

    def format_prompt(self, context: str, question: str) -> str:
        """Format the prompt with context and question"""
        return self.prompt.format(context=context, question=question)