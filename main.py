"""main module that has API defination along with the schemas"""
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from fastapi.responses import JSONResponse
import traceback
import logging
from vector_service import read_file
from vector_service import pinecone_service
from ai_service import gen_ai_functions
from ai_service import prompt
import config as ConfigTool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s -%(message)s')

load_dotenv()

origins = ['*']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = ConfigTool.get_config()
QUERY_PROMPT = prompt.QUERY_PROMPT

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request)-> JSONResponse:
    """This function is route of landing page

    Returns:
        JSONResponse: returns the status message for the initial connection
    """
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        error_message = type(e).__name__
        traceback_message = traceback.format_exc()
        logging.error("Error: %s\nTraceback: %s", error_message, traceback_message)
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


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(description="A file read as UploadFile")], question: Annotated[str, File(description="Question for the selected document")]) -> JSONResponse:
    """this api will take input from user and return the response from a source document
    Args:
        file: an input file
        question: a question to the input file
    Returns:
        Any: _description_
    """
    try:
        user_input = question
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
                "result": "Unsupported File Type, you can only choose '.txt', '.pdf', '.docx' or '.csv'"
            }
            return JSONResponse(content=message, status_code=409)
        if final_list:
            chain = gen_ai_functions.reduce_document_chain(config, QUERY_PROMPT)
            result = chain.invoke({"input_documents": final_list, "question": user_input})
            result = {
                "status":"Success",
                "result":result['output_text']
            }
            return JSONResponse(content=result)
        else:
            message =  {
                "status": "Failed",
                "result": "Some unexpected error occured while reading the file. That may be due to unsupported encoding standards."
            }
            return JSONResponse(content=message, status_code=409)
    except Exception as e:
       error_message = type(e).__name__
       traceback_message = traceback.format_exc()
       logging.error("Error: %s\nTraceback: %s", error_message, traceback_message)
       return JSONResponse(content={"status": "Failed", "result":"Some unexpected error occured!"}, status_code=409) 
