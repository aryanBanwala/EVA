import os
import uuid
import threading
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# in-memory buffers & lock
_POINTS_BUFFER: dict[str, list[PointStruct]] = {}
_BUFFER_LOCK    = threading.Lock()

# flush threshold
BUFFER_THRESHOLD = 3

def ensure_collection(collection_name, vector_dim: int):
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )


def flush_buffer(collection_name: str):
    """
    Thread-safely pull out whatever’s in the buffer and upload it.
    This function does NOT hold the lock during the upload itself.
    """
    # 1) Grab & clear under lock
    with _BUFFER_LOCK:
        pts = _POINTS_BUFFER.get(collection_name, [])
        if not pts:
            return
        # detach the list so new points can accumulate immediately
        _POINTS_BUFFER[collection_name] = []

    # 2) Upload outside the lock
    ensure_collection(collection_name, vector_dim=len(pts[0].vector))
    qdrant_client.upsert(collection_name=collection_name, points=pts)

def buffer_point(collection_name: str, vector: list[float], payload: dict):
    """
    Add to buffer, and if threshold is crossed, pull & upload
    WITHOUT holding the lock during the actual network call.
    """
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload or {})

    to_flush: list[PointStruct] | None = None
    with _BUFFER_LOCK:
        buf = _POINTS_BUFFER.setdefault(collection_name, [])
        buf.append(point)
        if len(buf) >= BUFFER_THRESHOLD:
            # swap out the list
            to_flush = buf.copy()
            _POINTS_BUFFER[collection_name] = []

    if to_flush:
        # upload outside the lock
        ensure_collection(collection_name, vector_dim=len(to_flush[0].vector))
        qdrant_client.upsert(collection_name=collection_name, points=to_flush)

    return point.id

def upload_embedding(collection_name, vector, payload=None, point_id=None):
    """Ensure collection and upload single vector with optional payload"""
    ensure_collection(collection_name, vector_dim=len(vector))

    if point_id is None:
        point_id = str(uuid.uuid4())

    point = PointStruct(
        id=point_id,
        vector=vector.tolist(),
        payload=payload or {}
    )

    qdrant_client.upsert(collection_name=collection_name, points=[point])
    # print(f"📤 Uploaded vector with ID: {point_id}")
    return point_id

def search_similar_vectors(collection_name, query_vector, top_k=5):
    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    return hits