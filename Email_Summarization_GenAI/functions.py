# Importing necessary libraries
import uvicorn
from contextlib import asynccontextmanager
import pickle
from fastapi import FastAPI,Request
from pydantic import BaseModel
import pickle
import pandas as pd
import re
import random
import logging
from openai import AzureOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import  RecursiveCharacterTextSplitter,  CharacterTextSplitter

from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredExcelLoader,  UnstructuredPowerPointLoader, Docx2txtLoader, PyPDFLoader,  UnstructuredWordDocumentLoader, TextLoader, UnstructuredFileLoader

from langchain.chains import RetrievalQA
from azure.core.credentials import AzureKeyCredential

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
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode
)

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

search_client = SearchClient(vector_store_endpoint, vector_index_name_resp, credential=AzureKeyCredential(vector_store_key))

#Defining a method to retrieve the relevant document from Azure AI Search
def Retrieve(Email):
    try:
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.85,"k": 5})
        docs = retriever.invoke(Email)
        if docs:
            return docs
        else:
            return("No relevant document found")
    except Exception as e:
        print(e) 
        return f"Error: {str(e)}"
    

#Defining a method to get the category of the given Email using relevant documents

def get_category(Email,Examples):
    try:
        if Examples == "No relevant document found":
            prompt = f"""Behave like a system and help me identify which can be the Primary Category of the given {Email} out of the listed categories given below:
            ['Clearance', 'Tracking', 'Other', 'Pickup', 'Customer Technology', 'Sales Support', 'Billing', 'Rating and Transit times', 'Dangerous Goods', 'Compliments and Complaints', 'GTS', 'Internal Escalations'].
            Instructions:
            1. Return Only Primary Category from the given list only. Don't create a different category on your own.
            2. Return just the name of Primary Category and nothing else.

            Output Format:
            Primary Category: #Mention Category#
            """
            result = llm_model.invoke(prompt)
            return result.content
        else:
            prompt = f"""Behave like a system. There are few examples of emails and it's Primary category given in {Examples}. Using this you have to predict what
            is going to be the Primary Category for {Email} out of the listed categories given below:
            ['Clearance', 'Tracking', 'Other', 'Pickup', 'Customer Technology', 'Sales Support', 'Billing', 'Rating and Transit times', 'Dangerous Goods', 'Compliments and Complaints', 'GTS', 'Internal Escalations'].

            Instructions:
            1. Return Only Primary Category and nothing else. Don't create a different category on your own other than the listed ones.

            Output Format:
            Primary Category: #Mention Category#
                    """
            result = llm_model.invoke(prompt)
            return result.content
    except Exception as e:
        print(str(e))
        return f"Error: {str(e)}"

def summarize(email):
    try:
        # Check if there are encrypted values in the email
        encrypted_texts = re.findall(r'<\[.*?\]>', email, flags=re.DOTALL)
        
        # Prepare the main content without encrypted texts
        main_content = re.sub(r'<\[.*?\]>', '', email, flags=re.DOTALL).strip()
        
        # Examples to illustrate the task
        examples = [
            {
                'original_body': "hi, we sent a package tracking <[uZSu3ODAj4nNtQ==]> 28&clienttype=ivother> 775873630828 on the 9th april which hasn't made it to its destination yet. it looks like it's at your aulnay sous bois depot in france for the last 10 days. from the tracking it appears that it cleared customs on the 11th april and delivery was attempted on the 12th. it says the delivery was refused but as it looks like it was 6:41 in the morning and i expect the business was closed. the package hasn't moved since. can you investigate and see why no attempt has been made to deliver the package since.",
                'summary': " Package tracking <[uZSu3ODAj4nNtQ==]> 28&clienttype=ivother> 775873630828 shows no delivery attempts since it was refused on the 12th April. Please investigate the reason for the delay."
            },
            {
                'original_body': "team, could you provide an update on order <[Gwl2SRRduniwuQ==]> with tracking 273574168819, which has shipped from the uk to us? there has been no update since 20th april and the ed delivery shows 22nd april 2024. the customer escalated earlier and we would need to ensure that it reaches them as promised. let us know how we can be of help to expedite this.",
                'summary': "Requesting an update on order <[Gwl2SRRduniwuQ==]> with tracking 273574168819, shipped from the UK to the US. No updates since April 20th, expected delivery on April 22nd, 2024. Customer escalated, need assistance to expedite delivery."
            },
            {
                'original_body': "your request <[9ciNqAul5w==]> has been updated. to add additional comments, reply to this email. george martin, apr 22, 2024, 08:29 gmt+1 fedex, can you advise as to why this item is being returned, can you also verify the below? - the return tracking number. - will this return be chargeable? thanks a lot for your help, i look forward to your response.",
                'summary': " Request <[9ciNqAul5w==]> updated. Need information from FedEx regarding return of item: return tracking number and if return is chargeable."
            },
            {
                'original_body': "i received a call this morning regarding shipment 136864862. contact <[cjNdRvAZWilCyISs6VpvYhvVHurvc2erg2Y=]> <[7eKR6VA6Ul/QH3eu677lDmDGe+DryP2xF/8=]> for customs clearance.",
                'summary': "Received call for shipment 136864862. Contact <[cjNdRvAZWilCyISs6VpvYhvVHurvc2erg2Y=]> <[7eKR6VA6Ul/QH3eu677lDmDGe+DryP2xF/8=]> for customs clearance."
            },
            {
                'original_body': "dear sian, i trust you are well. kindly escalate the below two orders - one is a delayed delivery and the other is a delayed return. please note this order was scheduled to be delivered on the 29th feb (yesterday) but there is a delayed status on this order - #1363 status – delayed tracking - 271413354269 <[Cz2ocIPWgaUNww==]> this order was requested to be returned but there has been a delay in the whole process - order - #1348 status - delayed return tracking - 775069656216 <[5sPV3R11NfpNpA==]>",
                'summary': "Please escalate two orders - one with a delayed delivery (#1363) and the other with a delayed return (#1348). The tracking numbers are 271413354269 <[Cz2ocIPWgaUNww==]> and 775069656216 <[5sPV3R11NfpNpA==]> respectively."
            },
            
            {
                'original_body': "hi team, i do not appreciate the lack of responses on your part. i wrote on the 12th of april and i have not heard back. please respond to my points below. please be reminded i expect full compensation from fedex due to the below. i am sorry to be badgering you about this but you are not responding to all my points (highlighted below) and you are not acknowledging the extent of the failures on your part. i already spoke with one of your representatives on this and they seem to have concurred with me that there have been some pretty glaring failures on fedex's part. 1. even if the third attempt was made at 8am and not 6am like the fedex app was showing, why was it not left at the shop, which was the pick up location? it was not my house/address - i had no power over picking up the package. i expected you to leave it there. this leaves one of two option",
                'summary': " The sender expresses frustration with the lack of response from the team regarding an email sent on April 12th. They expect full compensation from FedEx due to various failures on their part. One specific issue mentioned is the package not being left at the designated pick-up location."
            },
        ]
        
        # Randomly select an example email for guidance
        example_email = random.choice(examples)
        example_original_body = example_email['original_body'].strip()
        example_summary = example_email['summary'].strip()
        
        # Choose a random example sentence to prepend to the prompt
        example_sentence = random.choice([
            "Example: Please summarize the main points from this email according to the instructions given.",
            "Example: Summarize the following email into a brief summary according to the instructions given.",
            "Example: Create a concise summary of this email according to the instructions given."
        ])
        
        # Construct the prompt with examples integrated into the instructions
        if encrypted_texts:
            prompt = f"""The following examples show how to summarize emails with encrypted values:
            
            Example 1:
            Original: {example_original_body}
            Summary: {example_summary}
            
            Example 2:
            Original: {example_original_body}
            Summary: {example_summary}
            
            Now, summarize the following email:
            {main_content}
            
            Instructions:
            1. Include the encrypted values {encrypted_texts} in the summary which ever is important to create the summary.
            2. Keep the digits/numerical values from original email in the summary.
            3. Ensure to display the encrypted values {encrypted_texts} in the summary in the appropriate places as it is.
            4. Ensure the summary is concise and shorter than the original email.
            
            """
        else:
            prompt = f"""The following examples show how to summarize emails:

            Example 1:
            Original: {example_original_body}
            Summary: {example_summary}
            
            Example 2:
            Original: {example_original_body}
            Summary: {example_summary}
            
            Now, summarize the following email:
            {main_content}
            
            Instructions:
            1. Ensure the summary is concise and shorter than the original email.
            2. Keep the digits/numerical values from original email in the summary.
            3. Do not mention anything about encryption if no encrypted values are present.
            """
        
        # Invoke the language model with the refined prompt
        result = llm_model.invoke(prompt)
        return result.content.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"
    

def summarize_email_chain(EmailChain):
    prompt = f"""You are a bot responsible for summarizing email chains. Your task is to create a concise summary of the entire email chain. Follow these instructions:

    1. Read through the entire email chain and extract only the most important points, decisions, and actionable items.
    2. The summary should contain the imporant points from each email, so that it will give a good overview.
    3. Avoid including any details about dates, times, senders, recipients, or non-essential links and expressions.
    4. The summary should be brief and to the point, capturing the essence of the email chain without unnecessary details.
    5. Provide a concise paragraph that encapsulates the key issues and takeaways from the whole email chain.

    Email Chain: {EmailChain}
    """    
    result = llm_model.invoke(prompt)
    return result.content

def get_email_response(Email,Examples):
    try:
        if Examples == "No relevant document found":
            prompt = f"""Behave like a FedEx Customer Care Representative which assists the customers, understand who is the sender, understand the context of the email {Email} and help me write a humanly response to that sender with the following instructions:. 
            
            Instructions:
            1. Return the outgoing response with respect to the incoming email.
            2. The response should be short and on point.
            3. Show the concern according to the email body, no need to use apologies and excuses.
            4. Make sure to be polite and respectful towards the customer.
            5. Show concerns and empathy towards the customer regarding the issue discussed in {Email} but do not take control and declare the solution to the issue on your own.
            6. Generated response should be maximum 70-80 words long and not longer than that.

            
            Output Format:
            Outging Response: #outgoing email#
            """
            result = llm_model.invoke(prompt)
            return result.content
        else:
            prompt = f"""Behave like a FedEx Customer Care Representative which assists the customers, There are few examples of emails and it's outgoing responses given in {Examples}. Using this help me write a humanly response to that sender with the following instructions: 

            Instructions:
            1. Return the outgoing response to the incoming email only.
            2. Stick to the issue discussed in the incoming email.
            3. Show the concern according to the email body, no need to use apologies and excuses.
            4. Make sure to be polite and respectful towards the customer.
            5. Show concerns and empathy towards the customer regarding the issue discussed in {Email} but do not take control and declare the solution to the issue on your own.
            6. Generated response should be maximum 70-80 words long and not longer than that.

            Output Format:
            Outging Response: #outgoing email#
                    """
            result = llm_model.invoke(prompt)
            return result.content
    except Exception as e:
        return str(e)

def get_email_chain_response(email_chain):
    prompt = f""""Behave like a FedEx Customer Care Representative which assists the customers, understand the summary, understand the context of the whole email chain {email_chain} and help me write a humanly response to that sender with the following instructions:

    Instructions:
    1 Review the Entire Email Chain:
        Carefully read through the entire email chain to understand the context and key points,
        Identify the primary issue or request from the customer and any responses or actions taken so far.
    2 Summarize the Key Points:
        Note the critical issues raised by the customer,
        Highlight any actions already taken by FedEx or responses given in previous emails,
        Determine any unresolved concerns or requests from the customer.
    3 Draft the Response:
        Start with a polite and respectful greeting,
        Address the specific concern or request raised by the customer,
        Incorporate the essential details from the email chain to ensure the response is relevant,
        Show empathy and understanding towards the customer's situation, if applicable.
    4 Keep it Concise and On-Point:
        Ensure the response is focused and directly addresses the key points,
        Avoid including non-essential information, such as sender or recipient details, or irrelevant links,
        Aim for clarity and brevity, keeping the response within 70-80 words.
    5 Review and Revise:
        Re-read the response to ensure it meets all the instructions.
        Verify that it is polite, respectful, and empathetic where needed.
        Make sure it includes only the necessary context and maintains a professional tone.
        """
    result = llm_model.invoke(prompt)
    return result.content

def Retrieve_chunk_for_response(Email):
    try:
        query_vector = embeddings.embed_query(Email)
        vector_query = VectorizedQuery(vector=query_vector, 
                                    k_nearest_neighbors=7, 
                                    fields="incoming_email_vector")

        results = search_client.search(  
            search_text=None,  
            vector_queries=[vector_query],
            select=["incoming_email", "outgoing_email"],
            top=5
        )
        examples = []
        for result in results:  
            examples.append(result['outgoing_email'])
        return examples
    except Exception as e:
        print(e)
        return f"Error: {str(e)}"