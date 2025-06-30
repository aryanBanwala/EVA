import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, SearchRequest
import uuid

# Load environment variables
load_dotenv()

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Init client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ── Global buffers by collection ───────────────────────────────
_POINTS_BUFFER: dict[str, list[PointStruct]] = {}

def ensure_collection(collection_name, vector_dim=768):
    """Create collection if it doesn't exist"""
    collections = qdrant_client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        print(f"🧱 Creating collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    # else:
        # print(f"✅ Collection already exists: {collection_name}")

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

def buffer_point(collection_name: str, vector: list[float], payload: dict):
    """Create a PointStruct and store it in the in-memory buffer."""
    pts = _POINTS_BUFFER.setdefault(collection_name, [])
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload
    )
    pts.append(point)
    return point.id

def flush_buffer(collection_name: str):
    """Push all buffered points for this collection in one upsert call."""
    pts = _POINTS_BUFFER.get(collection_name)
    if not pts:
        return

    # ensure the collection exists (use first vector's length)
    ensure_collection(collection_name, vector_dim=len(pts[0].vector))

    qdrant_client.upsert(
        collection_name=collection_name,
        points=pts
    )
    # clear buffer after upload
    _POINTS_BUFFER[collection_name] = []