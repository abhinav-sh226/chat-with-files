from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from typing import Annotated
from fastapi.responses import JSONResponse
import traceback
import logging
import uvicorn
from vector_service import read_file
from vector_service import pinecone_service
from ai_service import helper_functions
import config as ConfigTool
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s -%(message)s')
# Load environment variables from .env file (if any)
load_dotenv()

class Response(BaseModel):
   user_input: str

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = ConfigTool.get_config()

@app.get("/")
async def home()-> JSONResponse:
    """This function is route of landing page

    Returns:
        JSONResponse: returns the status message
    """
    try:
        return JSONResponse(content={"status":"Success","message":"You are Connected successfully!"})
    except Exception as e:
        return JSONResponse(content={"status":"Failed","message":"Error Occured while creating the connection!"}, status_code=409)
        

@app.post("/upload_file")
async def upload_file(file: Annotated[UploadFile, File(description="A file read as UploadFile")])->JSONResponse:
    """The function will read and store the file to the PINECONE vector index

    Returns:
        JSONResponse: message and status after executing the API.
    """
    try:
        file_name = file.filename
        if file_name.endswith('.txt'):
            final_list =  await read_file.read_text_file(file)
        elif file_name.endswith('.pdf'):
            final_list =  await read_file.read_pdf_file(file)
        elif file_name.endswith('.csv'):
            final_list = await read_file.read_csv_file(file)
        elif file_name.endswith('.docx'):
            final_list = await read_file.read_docx_file(file)
        else:
            final_list = []
            message =  {
                "status": "Failed",
                "message": "Unsupported File Type, you can only choose '.txt', '.pdf', '.docx' or '.csv'"
            }
            return JSONResponse(content=message, status_code=409)
        if final_list:
            status = pinecone_service.ingest_document(config, final_list)
            if status["status"] == "Success":
                logging.info(status["message"])
            else:
                message =  {
                "status": "Failed",
                "message": status["message"]
            }
            return JSONResponse(content=message, status_code=409)
        else:
            message =  {
                "status": "Failed",
                "message": "Some unexpected error occured while reading the file. That may be due to unsupported encoding standards."
            }
            return JSONResponse(content=message, status_code=409)
    except Exception as e:
       error_message = type(e).__name__
       traceback_message = traceback.format_exc()
       logging.error("Error: %s\nTraceback: %s", error_message, traceback_message)
       return JSONResponse(content={"status": "Failed", "message":"Some unexpected error occured,"}, status_code=409) 


@app.post("/predict", response_model = Response)
async def predict(request: Response) -> JSONResponse:
    """this api will take input from user and return the response from a source document

    Args:
        request (Response): request from UI

    Returns:
        Any: _description_
    """
    query = request.user_input
    chain = helper_functions.get_qa_chain(config)
    result = chain(query)
    # #TODO: format the output as per user requirement
    return JSONResponse(content={"message":"Success"})
if __name__ == "__main__":
   uvicorn.run(app, host="localhost", port=8000)