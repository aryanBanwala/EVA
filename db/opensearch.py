import os
from uuid import uuid4
from dotenv import load_dotenv
from collections import defaultdict
from opensearchpy import OpenSearch, helpers

load_dotenv()

OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", 9200))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 512))

client = OpenSearch(
    hosts=[{'host': OS_HOST, 'port': OS_PORT}],
    use_ssl=False
)

BUFFER = defaultdict(list)

def ensure_index(index_name: str, dim: int = VECTOR_DIM):
    if not client.indices.exists(index=index_name):
        print("index doesnt exist")
        
def upload_embedding(index_name: str, vector: list[float], payload: dict = None, doc_id: str = None):
    ensure_index(index_name, dim=len(vector))
    body = {
        "embedding": vector,
        **(payload or {})
    }
    client.index(index=index_name, id=doc_id or str(uuid4()), body=body)

def search_similar_vectors(index_name: str, query_vector, top_k: int = 5):
    # tensor → list
    vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

    body = {
        "size": top_k,
        "query": {
            "knn": {                 
                "embedding": {       
                    "vector": vector,
                    "k": top_k
                }
            }
        }
    }

    res = client.search(index=index_name, body=body)

    # wrap hits like Qdrant
    hits = [
        type("Hit", (), {
            "id":      h["_id"],
            "score":   h["_score"],
            "payload": h["_source"]
        })()
        for h in res["hits"]["hits"]
    ]
    return hits

def buffer_point(index_name: str, vector: list[float], payload: dict = None):
    doc_id = str(uuid4())
    doc = {
        "_index": index_name,
        "_id": doc_id,
        "_source": {
            "embedding": vector,
            **(payload or {})
        }
    }
    BUFFER[index_name].append(doc)
    return doc_id

def flush_buffer(index_name: str):
    if not BUFFER[index_name]:
        return
    ensure_index(index_name, dim=len(BUFFER[index_name][0]['_source']['embedding']))
    helpers.bulk(client, BUFFER[index_name])
    print(f"📤 Flushed {len(BUFFER[index_name])} vectors to {index_name}")
    BUFFER[index_name] = []

def delete_index(index_name: str):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"🗑️ Deleted index: {index_name}")