import time
from typing import List
from bson.objectid import ObjectId
from db.mongo.connection import get_db

_feeds_collection = None

def init():
    """
    Initialize the feeds collection instance.
    Call this once (sync) after init_connection().
    """
    global _feeds_collection
    if _feeds_collection is not None:
        return
    _feeds_collection = get_db()["feeds"]
    print("[MongoDB] Feeds collection initialized")

def _get():
    if _feeds_collection is None:
        raise Exception("Feeds not initialized. Call init() first.")
    return _feeds_collection

# Shared filter for single-video media not in blacklist
common_media_filter = {
    "data.media": {
        "$elemMatch": {
            "type": "video",
            "url": {
                "$nin": [
                    "post/default/media-under-process.mp4",
                    "post/61cc46d333989b18bcfc9dfb/HLS/static/media-under-process.mp4"
                ]
            }
        }
    }
}

async def fetch_video_feeds_to_embed_backlog(page: int = 1, limit: int = 10):
    now = int(time.time())
    ten_months_ago = now - (10 * 30 * 24 * 3600)
    one_hour_ago   = now - 3600
    skip = (page - 1) * limit

    pipeline = [
        { "$match": {
            "vectorModelStatus": { "$exists": False },
            "created_at": { "$gte": ten_months_ago, "$lt": one_hour_ago },
            "content_type": "video"
        }},
        { "$match": { "data.media": { "$size": 1 } }},
        { "$match": common_media_filter },
        { "$sort": { "_id": -1 }},
        { "$skip": skip },
        { "$limit": limit }
    ]

    cursor = _get().aggregate(pipeline)
    results = []
    async for doc in cursor:
        results.append({
            "feedId": str(doc["_id"]),
            "author": doc["author"],
            "url": doc["data"]["media"][0]["url"],
            "created_at": doc["created_at"]
        })
    return results

async def fetch_video_feeds_to_embed_one_hour(page: int = 1, limit: int = 10):
    one_hour_ago = int(time.time()) - 3600
    skip = (page - 1) * limit

    pipeline = [
        { "$match": {
            "vectorModelStatus": { "$exists": False },
            "created_at": { "$gte": one_hour_ago },
            "content_type": "video",
            "author": { "$ne": "67e7d26e054269544b7f9809" }
        }},
        { "$match": { "data.media": { "$size": 1 } }},
        { "$match": common_media_filter },
        { "$sort": { "_id": -1 }},
        { "$skip": skip },
        { "$limit": limit }
    ]

    cursor = _get().aggregate(pipeline)
    results = []
    async for doc in cursor:
        results.append({
            "feedId": str(doc["_id"]),
            "author": doc["author"],
            "url": doc["data"]["media"][0]["url"],
            "created_at": doc["created_at"]
        })
    return results

async def update_vector_model_status(ids: List[str], status: int) -> int:
    """
    Bulk-update vectorModelStatus for a list of feed IDs.
    Returns the number of documents modified.
    """
    object_ids = [ObjectId(i) for i in ids]
    result = await _get().update_many(
        { "_id": { "$in": object_ids } },
        { "$set": { "vectorModelStatus": status } }
    )
    return result.modified_count