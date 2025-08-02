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
        host = os.getenv("MONGO_HOST")
        user = os.getenv("MONGO_USER")
        password = os.getenv("MONGO_PASS")
        db_name = os.getenv("MONGO_DB")

        if not all([host, user, password, db_name]):
            raise Exception("❌ Missing MongoDB PROD credentials")

        uri = f"mongodb+srv://{user}:{password}@{host}/{db_name}?retryWrites=true&w=majority"
        print(f"[MongoDB:PROD] Connecting to Atlas: {host} (DB: {db_name})")

        _client = AsyncIOMotorClient(uri)
        _db = _client[db_name]

    else:
        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", 27017))
        db_name = os.getenv("MONGO_DB", "build-socialMedia-stag")

        print(f"[MongoDB:DEV] Connecting to {host}:{port} (DB: {db_name})")

        _client = AsyncIOMotorClient(host=host, port=port)
        _db = _client[db_name]

    try:
        await _client.admin.command("ping")
        print("[MongoDB] ✅ Connected successfully")
    except Exception as e:
        print(f"[MongoDB] ❌ Connection failed: {e}")
        raise

def get_db():
    if _db is None:
        raise Exception("MongoDB not initialized. Call init_connection() first.")
    return _db