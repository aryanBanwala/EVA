# connection.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("_ENV", "local")

_client = None
_db = None

async def init_connection():
    global _client, _db
    
    if _client is not None and _db is not None:
        return

    if ENV == "prod":
        host = os.getenv("MONGO_HOST", "mongo.prod.internal.com")
        port = int(os.getenv("MONGO_PORT", 27017))
        db_name = os.getenv("MONGO_DB", "socialMedia-prod")
    else:
        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", 27017))
        db_name = os.getenv("MONGO_DB", "build-socialMedia-stag")

    print(f"[MongoDB] Connecting to {host}:{port} (DB: {db_name})")
    try:
        _client = AsyncIOMotorClient(host=host, port=port)
        _db = _client[db_name]
        await _client.admin.command('ping')
        print("[MongoDB] ✅ Connected successfully")
    except Exception as e:
        print(f"[MongoDB] ❌ Connection failed: {e}")
        raise


def get_db():
    if _db is None:
        raise Exception("MongoDB not initialized. Call init_connection() first.")
    return _db