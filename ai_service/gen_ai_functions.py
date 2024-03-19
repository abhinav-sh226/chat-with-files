"""This module contains the frequently used functions"""
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ReduceDocumentsChain, RetrievalQA, RetrievalQAWithSourcesChain)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import BasePromptTemplate
from vector_service import pinecone_service
from ai_service import prompt

DOCUMENT_PROMPT = prompt.DOCUMENT_PROMPT
COLLAPSE_PROMPT = prompt.COLLAPSE_PROMPT

def get_embeddings(config: dict) -> OpenAIEmbeddings:
    """the function returns the embedding model

    Args:
        config (dict): credentials

    Returns:
        OpenAIEmbeddings: the embedding model
    """
    embedding_model = OpenAIEmbeddings(
        openai_api_key = config["openai_key"],
        model="text-embedding-3-small",
        max_retries=4,
        chunk_size=1000,
        retry_min_seconds=10,
        retry_max_seconds=30
        )
    return embedding_model


def get_llm_model(config: dict) -> ChatOpenAI:
    """return the llm

    Args:
        config (dict): credentials

    Returns:
        ChatOpenAI: llm
    """
    llm = ChatOpenAI(
    openai_api_key = config["openai_key"],
    model_name = 'gpt-3.5-turbo',
    temperature = 0.0
    )
    return llm


def get_vector_store(config):
    pc = pinecone_service.connect_pinecone(config)
    text_field = "text"
    index = config["index_name"]
    embeddings = get_embeddings(config)
    vectorstore = PineconeVectorStore(
        index, embeddings, text_field
    ) 
    return vectorstore


def get_qa_chain(config: dict) -> RetrievalQAWithSourcesChain:
    """Function return a chain

    Args:
        config (dict): credentials

    Returns:
        RetrievalQAWithSourcesChain: chain
    """
    llm = get_llm_model(config)
    vectorstore = get_vector_store(config)
    qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )
    return qa


def get_llm_chain(config: dict, input_prompt: BasePromptTemplate) -> LLMChain:
    """function return a chain

    Args:
        config (dict): credentials
        input_prompt (BasePromptTemplate): prompt to the chain
    Returns:
        LLMChain: chain from langchain
    """
    llm = get_llm_model(config)
    llm_chain = LLMChain(
        llm=llm,
        prompt=input_prompt,
        verbose=True
    )
    return llm_chain


def reduce_document_chain(config: dict, prompt: BasePromptTemplate)-> ReduceDocumentsChain:
    """Return a chain that reduce the chance of token limit exceeded error

    Args:
        config (dict): configuration
        prompt (BasePromptTemplate): prompt to the ReduceDocumentsChain

    Returns:
        ReduceDocumentsChain: chain
    """
    document_variable_name = "documents"
    llm_chain = get_llm_chain(config, input_prompt = prompt)
    combine_documnets_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=DOCUMENT_PROMPT,
        document_variable_name=document_variable_name
    )
    llm_chain = get_llm_chain(config, input_prompt = COLLAPSE_PROMPT)
    collapse_document_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=DOCUMENT_PROMPT,
        document_variable_name=document_variable_name
    )
    chain = ReduceDocumentsChain(
        combine_documents_chain = combine_documnets_chain,
        collapse_documents_chain = collapse_document_chain
    )
    return chain
