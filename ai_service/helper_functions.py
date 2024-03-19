"""This module contains the frequently used functions"""
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_pinecone import PineconeVectorStore
from vector_service import pinecone_service

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
        chunk_size=200,
        retry_min_seconds=10,
        retry_max_seconds=30
        )
    return embedding_model

def get_llm_model(config: dict) -> ChatOpenAI:
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
    llm = get_llm_model(config)
    vectorstore = get_vector_store(config)
    qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )
    return qa

