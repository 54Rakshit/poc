import os
import redis

# from langchain.vectorstores.redis import Redis
from redisvl.extensions.llmcache import SemanticCache
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import logging
from dotenv import load_dotenv
from utils.chunking import create_redisvl_schema,chunking,customVectorQuery

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Environment variables
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_INDEX = os.getenv("REDIS_INDEX")


def create_redis_client(host, port, password):
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)

def create_redis_index(client):
    schema = [
        TextField("query"),
        VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"})
    ]
    try:
        client.ft(REDIS_INDEX).create_index(schema, definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.JSON))
        logger.info("Redis index created successfully.")
    except redis.exceptions.ResponseError as e:
        if "Index already exists" in str(e):
            logger.info("Index already exists.")
        else:
            logger.error(f"Error creating Redis index: {e}")

def get_from_cache(query):
    try:
        # client = create_redis_client(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)
        from .storage import get_all_blob_data
        documents = get_all_blob_data(AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER_NAME, CHAT_HISTORY_BLOB)
        if documents:
            client = create_redis_client(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)
            if(client.ping()==True):
                print("client ping successful in get_from_cache")
                cached_data = semantic_search(query,client,documents)
            else:
                print("client was not created in get_from_cache")
            #logger.info(f"cached_data",cached_data)
            print(f"....................cached data is coming?", cached_data[0:200])
            logger.info("Cache hit. Returning cached response.")
            return cached_data
        return []
    except Exception as e:
        logger.error(f"Error getting from cache: {e}")
    return None

def semantic_search(query,client,documents):
    try:
        print("inside semantic_search function")
        if(client):
            print("client is available in semantic_search")
            create_redisvl_schema(client)
        else:
            print("client not present in sementic search")
        chunking(documents)
        result = customVectorQuery(query)
        return result
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        return []
    
def update_cache(query, response):
    try:
        client = create_redis_client(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)
        semantic_cache = SemanticCache(client, index_name=REDIS_INDEX)
        semantic_cache.add({"query": query, "response": response})
        logger.info("Cache updated successfully.")
    except Exception as e:
        logger.error(f"Error updating cache: {e}")
