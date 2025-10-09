# Importing necessary libraries
import uvicorn
from contextlib import asynccontextmanager
import pickle
from azure.core.credentials import AzureKeyCredential
from fastapi import FastAPI,Request
from pydantic import BaseModel
import pickle
import pandas as pd
import re
import random
import logging
from openai import AzureOpenAI
from fastapi import FastAPI, Body, HTTPException
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import  RecursiveCharacterTextSplitter,  CharacterTextSplitter

from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredExcelLoader,  UnstructuredPowerPointLoader, Docx2txtLoader, PyPDFLoader,  UnstructuredWordDocumentLoader, TextLoader, UnstructuredFileLoader

from langchain.chains import RetrievalQA

from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,
    SearchField,  
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
) 

from functions import Retrieve, get_category, summarize, summarize_email_chain, get_email_chain_response, get_email_response, Retrieve_chunk_for_response
from cleaning import clean_email

logger = logging.getLogger(__name__)

# Defining a class for Tariff code prediction
class Data(BaseModel):

    EmailText: str

#Initiating the FastAPI
app = FastAPI(
    title="Email Category Classification",
    description="A simple API that is used to classify the category of Email ",
    version="0.1",
    debug=True
)

#Declaring the variables and environment
open_ai_type = "azure"
open_ai_api_version = "2023-05-15"
open_ai_api_key = "open_ai_api_key"
open_ai_api_key = open_ai_api_key 
open_ai_endpoint = "open_ai_endpoint" # base
embeddings_model = "janus-text-embedding-ada-002"
gpt_deployment_4 = "janus-gpt-4"
gpt_deployment_35 = "janus-gpt-35-turbo"
gpt_deployment_35_16="janus-gpt-35-turbo-16k"
embeddings_model = "janus-text-embedding-ada-002"
vector_store_endpoint = "vector_store_endpoint"
vector_store_key = "vector_store_key"
vector_index_name_category = "vector_index_name_category"
vector_index_name_resp = "vector_index_name_resp"

#Initiating the LLM, Embedding and Azure AI Search
deployment_name = gpt_deployment_35
llm_model = AzureChatOpenAI(deployment_name=deployment_name, openai_api_key =open_ai_api_key, azure_endpoint = open_ai_endpoint, openai_api_version = open_ai_api_version, temperature=0) 

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_model,
    openai_api_version=open_ai_api_version,
    azure_endpoint = open_ai_endpoint,
    openai_api_key = open_ai_api_key
)

vector_store = AzureSearch(
    azure_search_endpoint=vector_store_endpoint ,
    azure_search_key=vector_store_key,
    index_name=vector_index_name_category,
    embedding_function=embeddings.embed_query
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("lifespan start")
    yield
    # Clean up the ML models and release the resources
    print("lifespan end")

# app = FastAPI(lifespan=lifespan)
app = FastAPI(lifespan=lifespan,
    title="Email Category Classification",
    description="A simple API that is used to classify the category of Email ",
    version="0.1",
    debug=True
)

@app.get('/')

def index():
    return {'message': 'This is the homepage of the API '}

@app.middleware("http")
async def access_logger_middleware(request: Request, call_next):
    response = await call_next(request)
    msg = "{host}:{port} {method} {path} {status_code}".format(host=request.client.host, port=request.client.port,
        method=request.method, path=request.url, status_code=response.status_code)
    logger.info(msg)
    return response

@app.post('/Predict_primary_category')
def get_primary_category(text: str = Body(..., media_type="text/plain")):
    try:
        data = Data(EmailText=text)
        email_text = data.EmailText
        text = clean_email(email_text)
        document = Retrieve(text)
        PC = get_category(text,document)
        # Summary = summarize(data.EmailText)
        return  PC
    except Exception as e:
        print(e)
        logger.exception(e)
        return f"Error: {str(e)}"
    
@app.post('/Get_summary')
def get_summary(text: str = Body(..., media_type="text/plain")):
    try:
        data = Data(EmailText=text)
        email_text = data.EmailText
        text = clean_email(email_text)
        Summary = summarize(text)
        return  Summary
    except Exception as e:
        print(e)
        logger.exception(e)
        return f"Error: {str(e)}"
    
@app.post('/Get_auto_response')
def get_auto_resp(text: str = Body(..., media_type="text/plain")):
    try:
        data = Data(EmailText=text)
        email_text = data.EmailText
        text = clean_email(email_text)
        examples = Retrieve_chunk_for_response(text)
        resp = get_email_response(text,examples)
        return  resp
    except Exception as e:
        print(e)
        logger.exception(e)
        return f"Error: {str(e)}"
    
@app.post('/Summarize_email_chain')
def get_emailchain_summary(text: str = Body(..., media_type="text/plain")):
    try:
        data = Data(EmailText=text)
        email_text = data.EmailText
        email_chain = clean_email(email_text)
        Summary = summarize_email_chain(email_chain)
        return  Summary
    except Exception as e:
        print(e)
        logger.exception(e)
        return f"Error: {str(e)}"    
    
@app.post('/Get_email_chain_auto_response')
def get_auto_resp_email_chain(text: str = Body(..., media_type="text/plain")):
    try:
        data = Data(EmailText=text)
        email_text = data.EmailText
        email_chain = clean_email(email_text)
        resp = get_email_chain_response(email_chain)
        return  resp
    except Exception as e:
        print(e)
        logger.exception(e)
        return f"Error: {str(e)}"    

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=7000)