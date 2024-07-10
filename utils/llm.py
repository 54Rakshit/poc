# import requests
from utils.storage import get_all_blob_data
import logging
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
# import numpy as np
from utils.chunking import chunking, create_redisvl_schema
from utils.cache import create_redis_client
# from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# from torch.utils.data import Dataset

load_dotenv()

# Environment Variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
HR_POLICIES_BLOB = os.getenv("HR_POLICIES_BLOB") 
deployment_name = os.getenv("deployment_name")
model_name = os.getenv("model_name")
REDIS_HOST = os.getenv("REDIS_HOST") 
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the AzureChatOpenAI model
llm = AzureChatOpenAI(
    temperature=0.2,
    deployment_name=deployment_name,
    model_name=model_name
)

def query_llm(query, context):
    logger.info(f"Query: {query}")
    logger.info(f"Context: {context}")

    template = """
    You are a helpful and knowledgeable HR assistant. Below is some context that might help you answer the query accurately.

    Context: {context}

    Please answer the following query based on the provided context. Be precise, clear, and professional in your response. If the context does not provide enough information to answer the query, indicate that and suggest contacting HR for further assistance.

    Query: {query}
    """
    prompt = PromptTemplate(input_variables=["query", "context"], template=template)
    output_parser = StrOutputParser()

    # Combine the prompt and the language model
    chain = prompt | llm | output_parser
    
    try:
        # Invoke the chain with the input variables
        response = chain.invoke({"query": query, "context": context})
        logger.info("Response: %s", response)
        return response
    except Exception as e:
        logger.error("Error querying LLM: %s", str(e))
        return None

def train_llm(endpoint, api_key, connection_string, container_name, blob_names):
    try:
        for blobname in blob_names:
            documents = get_all_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, blobname)
            # print("checking document coming?", documents)
        if documents:
            logger.info("Blob data retrieved successfully.")
            
            # Chunk documents and store embeddings
            client = create_redis_client(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)
            create_redisvl_schema(client)
            chunking(documents)
            print("chunking successful")
            # with open('embeddings.npy', 'wb') as f:
            #     np.save(f, embeddings)
            # # Load embeddings for training
            # embeddings = np.load('embeddings.npy')
                
            # yet to write training logic
            # # Load tokenizer and model
            # model_name = model_name  # Update to GPT-4
            # model = AutoModelForCausalLM.from_pretrained(model_name)
            # tokenizer = AutoTokenizer.from_pretrained(model_name)

            # # Create the training dataset
            # train_dataset = CustomDataset(documents, tokenizer)

            # # Set up training arguments
            # training_args = TrainingArguments(
            #     output_dir="./results",  # output directory
            #     num_train_epochs=1,  # number of training epochs
            #     per_device_train_batch_size=4,  # batch size for training
            #     save_steps=10_000,  # number of updates steps before checkpoint saves
            #     save_total_limit=2,  # limit the total amount of checkpoints
            # )

            # # Initialize the Trainer
            # trainer = Trainer(
            #     model=model,
            #     args=training_args,
            #     train_dataset=train_dataset,
            # )

            # # Start training
            # trainer.train()
                
            logger.info("LLM training complete.")
        else:
            logger.warning("No documents found for the provided blob names.")
    except Exception as e:
        logger.error(f"Error in training LLM: {e}")

# class CustomDataset(Dataset):
#     def __init__(self, texts, tokenizer):
#         self.tokenizer = tokenizer
#         self.texts = texts

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         # Tokenize the text data for training
#         encoding = self.tokenizer(self.texts[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
#         # Flatten the dictionary to handle DataLoader
#         return {key: tensor.squeeze() for key, tensor in encoding.items()}