"""This module belong to all the service related to pinecone vector db"""
from pinecone import Pinecone, PodSpec
from langchain_pinecone import PineconeVectorStore
from ai_service import gen_ai_functions
import logging
import traceback
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s -%(message)s')

def connect_pinecone(config)->None:
    """Creates a connection with the pinecone

    Args:
        config (_type_): _description_
    """
    try:
        pc = Pinecone(api_key = config["pinecone_api_key"])
        logging.info("Connected with the Pinecone Successfully!")
        return pc
    except Exception as e:
       error_message = type(e).__name__
       traceback_message = traceback.format_exc()
       logging.error("Error: %s\nTraceback: %s", error_message, traceback_message)


def ingest_document(config, content_list):
    try:
        embeddings = gen_ai_functions.get_embeddings(config)
        pc = connect_pinecone(config)
        index_name = config["index_name"]
        if index_name in pc.list_indexes().names():
            logging.info("Index already exist...")
            logging.info("Deleting the index...")
            pc.delete_index(index_name)    
        else: 
            logging.info("Index does not exist...")
        logging.info("Creating the index...")
        pc.create_index(
            index_name,
            dimension=config["index_dimension"],
            metric='dotproduct',
            spec = PodSpec(config["pinecone_environment"])
        )
        PineconeVectorStore.from_documents(content_list, embeddings, index_name=index_name)
        message = {
            "status": "Success",
            "message": "Documents ingested successfully"
        }
        return message
    except Exception as e:
       error_message = type(e).__name__
       traceback_message = traceback.format_exc()
       logging.error("Error: %s\nTraceback: %s", error_message, traceback_message)
       message = {
            "status": "Failed",
            "message": "Documents ingestion Failed"
        }
       return message
