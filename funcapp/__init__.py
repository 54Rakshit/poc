from flask import Flask, request, jsonify
import os
from utils.cache import get_from_cache, update_cache
from utils.storage import get_relevant_blob_data, get_updated_blobs, get_all_blob_data
from utils.llm import query_llm, train_llm
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

#Environment variables
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_INDEX = os.getenv("REDIS_INDEX")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
HR_POLICIES_BLOB = os.getenv("HR_POLICIES_BLOB")
CHAT_HISTORY_BLOB = os.getenv("CHAT_HISTORY_BLOB")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

@app.route("/pretrain_llm", methods=['POST'])
def pretrain_llm_endpoint():
    blobs = [HR_POLICIES_BLOB, CHAT_HISTORY_BLOB]
    train_llm(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, blobs)
    if train_llm:
        return jsonify({"status": "LLM pretrained"})
    else:
        return jsonify({"status": "LLM not pretrained"})

@app.route("/query", methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    response = get_from_cache(query)
    if not response:
        relevant_data = get_relevant_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, HR_POLICIES_BLOB, query)
        print("inside main printing relevant_data")
        print(relevant_data[0:10])
        response = query_llm(query, relevant_data)
        # update_cache(query, response)
    return jsonify({"response": response})

@app.route("/update_cache", methods=['POST'])
def update_cache_endpoint():
    updated_blobs = get_updated_blobs(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, [CHAT_HISTORY_BLOB])
    if updated_blobs:
        for blob in updated_blobs:
            data = get_all_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, [blob])
            for item in data:
                update_cache(item['query'], item['response'])
    return jsonify({"status": "Cache updated"})

@app.route("/retrain_llm", methods=['POST'])
def retrain_llm_endpoint():
    updated_blobs = get_updated_blobs(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, [HR_POLICIES_BLOB])
    if updated_blobs:
        train_llm(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, updated_blobs)
    return jsonify({"status": "LLM retrained"})

if __name__ == '__main__':
    app.run(debug=True)
