"""This module has all the configurations"""
import os
from dotenv import load_dotenv

def get_config() -> dict:
    """this function will return the configurations

    Returns:
        config (dict): dict containing all the configurations
    """
    load_dotenv("./.env")
    config = {}
    config["pinecone_environment"] = os.getenv("PINECONE_ENV")
    config["pinecone_api_key"] = os.getenv("PINECONE_API_KEY")
    config["index_name"] = "sample-index"
    config["index_dimension"] = 1536 #as per text-embedding-ada-002
    config["openai_key"] = os.getenv("OPENAI_API_KEY")
    return config
