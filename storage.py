import logging
import os
# import numpy as np
from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF for reading PDF files
import docx
from dotenv import load_dotenv
from utils.cache import create_redis_client, semantic_search
from docx import Document

load_dotenv()

# Environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
HR_POLICIES_BLOB = os.getenv("HR_POLICIES_BLOB")  
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_INDEX = os.getenv("REDIS_INDEX")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_blob_data(connection_string, container_name, folder_path):
    try:
        # initialize the connection
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        logger.info("Listing blobs...")
        blobs_list = container_client.list_blobs(name_starts_with=folder_path) #folders

        blobs = [] #folder/file names
        for blob in blobs_list: #folder/file name
            print(blob.name) #folder/file name
            blobs.append(blob.name) #folder/file name

        logger.info("Blobs: %s", blobs)

        documents = {}

        for blob_name in blobs: #folder/file name in folders
            blob_client = container_client.get_blob_client(blob_name) #folder/file name

            # Read the blob's content into memory
            logger.info(f"Reading blob: {blob_name}") #folder
            blob_data = blob_client.download_blob().readall() #folder data
            # print(blob_data)
            file_path = os.path.join("/tmp", os.path.basename(blob_name))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                file.write(blob_data)

            # Read file content based on the file type
            if blob_name.endswith('.pdf'):
                documents[blob_name] = read_pdf(file_path)
            elif blob_name.endswith('.txt'):
                documents[blob_name] = read_txt(file_path)
            elif blob_name.endswith('.docx'):
                documents[blob_name] = read_docx(file_path)
            else:
                logger.warning(f"Skipping unsupported file type: {blob_name}")

        logger.info("Blob data retrieval successful.")
        return documents

    except Exception as e:
        logger.error(f"Error getting blob data: {e}")
        return {}

def read_txt(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
        # print(f"Page {page_num + 1}:\n{text}")
    return text

def read_docx(file_path):
    print("checking if able to reach doc function")
    print(file_path)
    doci = Document(file_path)
    print("checking if able to reach doc file", doci)
    text = "\n".join([paragraph.text for paragraph in doci.paragraphs])
    return text

def get_relevant_blob_data(connection_string, container_name, folder_path, query):
    documents = get_all_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, HR_POLICIES_BLOB)
    print("Inside get_relevant_blob_data")
    if documents:
        client = create_redis_client(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)
        if(client.ping()==True):
            print("client ping successful in get_relavant_blob_data")
            relevant_data = semantic_search(query,client,documents)
        else:
            print("client was not created in get_relavant_blob_data")
        #logger.info(f"relevant_data",relevant_data)
        print(f"relevant data is coming?", relevant_data[0:2])
        return relevant_data
    return []

def get_updated_blobs(connection_string, container_name, blobs):
    try:
        logger.info("Retrieving updated blobs.")
        updated_blobs = blobs
        return updated_blobs
    except Exception as e:
        logger.error(f"Error getting updated blobs: {e}")
        return []
