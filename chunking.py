import os
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer
from dotenv import load_dotenv
# from utils.storage import get_all_blob_data
import logging 

#create iterate dict
from redisvl.redis.utils import array_to_buffer

#VectorQuery
from redisvl.query import VectorQuery

load_dotenv()
#Environment variables
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_INDEX = os.getenv("REDIS_INDEX")
REDIS_URL = os.getenv("REDIS_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME= os.getenv("BLOB_CONTAINER_NAME")
HR_POLICIES_BLOB = os.getenv("HR_POLICIES_BLOB")
CHAT_HISTORY_BLOB = os.getenv("CHAT_HISTORY_BLOB")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2")
client = 1
index = 2

def create_redisvl_schema(client): 
    global index
    global hf
    # print("inside create_redisvl_schema function")
    # print(REDIS_INDEX)
    # print(hf.dims)
    schema = IndexSchema.from_dict({
    "index": {
        "name": REDIS_INDEX,
        "prefix": "chunk"
    },
    "fields": [
        {
            "name": "doc_id",
            "type": "tag",
            "attrs": {
                "sortable": True
            }
        },
        {
            "name": "content",
            "type": "text"
        },
        {
            "name": "text_embedding",
            "type": "vector",
            "attrs": {
                "dims": hf.dims,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32"
            }
        }
    ]
    })
    # print("printing schema")
    # print(schema)
    # connect to redis
    print(client.ping())
    # create an index from schema and the client
    index = SearchIndex(schema, client)
    index.create(overwrite=True, drop=True)

def chunking(chunk_dict):
    global index
    # print("index checking" , index)
    global hf
    # print("index checking" , hf)
    # print("inside chunking function")

    try:        
        #create chunks
        chunks = []
        for key, value in chunk_dict.items():
            for i in range(0, len(value), 2500):
                chunks.append(value[i:i + 2500])
        
        print(chunks[0])
        # print(chunks[1])
        # print(len(chunks))
        # print(type(chunks[0]))
        # if (type(chunks[:]) != str):
        #     chunks = [chunk.decode('utf-8',  errors='ignore') for chunk in chunks]
        # print(type(chunks[0]))
        # print(type(chunks[0]))
        chunks = convert_chunks_to_str(chunks)
        # chunks = [str(chunk) for chunk in converted_chunks]
        # print(chunks) 
        # print(type(chunks[0]))
        #create embeddings
        embeddings = hf.embed_many(chunks)
        # print(embeddings)
        # print(len(embeddings))
        if(len(embeddings)==len(chunks)):
            logger.info("Blob data retrieval successful.")
        
        # load expects an iterable of dictionaries
        i = 0
        data = []
        for chunk in chunks:
            temp_dict = {
                'doc_id': f'{i}',
                'content': chunk,
                # For HASH -- must convert embeddings to bytes
                'text_embedding': array_to_buffer(embeddings[i])
            }
            data.append(temp_dict)
            i = i + 1
        # print(data)
        # RedisVL handles batching automatically
        keys = index.load(data, id_field="doc_id")
        # print(keys)
    except Exception as e:
        logger.error(f"Error getting blob data: {e}")
        return {}

def convert_chunks_to_str(chunks):
    converted_chunks = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            try:
                converted_chunks.append(chunk.decode('utf-8'))
            except UnicodeDecodeError:
                converted_chunks.append(str(chunk))  # Fallback to string representation of bytes
        else:
            converted_chunks.append(str(chunk))  # Convert other types to string if necessary
    return converted_chunks

def customVectorQuery(query):
    global index
    global hf
    #process query
    # print("inside CustomVectorQuery function")
    # print("printing query")
    # print(query)
    query_embedding = hf.embed(query)
    vector_query = VectorQuery(
        vector=query_embedding,
        vector_field_name="text_embedding",
        num_results=1,
        return_fields=["doc_id", "content"],
        return_score=True
    )

    #semantic search query on index which contains data()
    combined_result = index.query(vector_query)
    result_content = combined_result[0]["content"]
    return result_content