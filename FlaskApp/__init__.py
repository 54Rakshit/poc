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



@app.route("/")
def index():
    return (
        "Try /hello/Vijay for parameterized Flask route.\n"
        "Try /module for module import guidance"
    )

@app.route("/hello/<name>", methods=['GET'])
def hello(name: str):
    return f"hello {name}"


@app.route("/query", methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    response = get_from_cache(query)
    if not response:
        relevant_data = get_relevant_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, HR_POLICIES_BLOB, query)
        print("pragya : inside main printing relevant_data")
        print(relevant_data[0:10])
        response = query_llm(query, relevant_data)
        # update_cache(query, response)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
