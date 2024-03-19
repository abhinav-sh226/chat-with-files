"""This module has all the prompts used in the project"""
from langchain_core.prompts import PromptTemplate

DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["documents"],
    template="{page_content}"
)


COLLAPSE_PROMPT = PromptTemplate.from_template(
    "Collapse this content: {documents}"
)


QUERY_PROMPT = PromptTemplate.from_template(
    """You will be give a Document and a Question.
       You need to answer the question from the given document.
       Document: {documents}
       Question: {question}
    """
)
