# vector.py
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers, RequestsHttpConnection

load_dotenv(override=True)

ENV = os.getenv("_ENV", "local")

if ENV == "prod":
    OS_HOST = os.getenv("OS_HOST", "prod-es.internal.com")
    OS_PORT = int(os.getenv("OS_PORT", "9200"))
    OS_INDEX = os.getenv("OS_INDEX", "prod-vectors")

    client = OpenSearch(
        hosts=[{"host": OS_HOST, "port": OS_PORT}],
        http_compress=True,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

else:
    OS_HOST = os.getenv("OS_HOST", "localhost")
    OS_PORT = int(os.getenv("OS_PORT", "9200"))
    OS_INDEX = os.getenv("OS_INDEX", "test-vectors")

    client = OpenSearch(
        hosts=[{"host": OS_HOST, "port": OS_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )


def bulk_push_vector_docs(docs):
    actions = []
    for doc in docs:
        actions.append({
            "_op_type": "index",
            "_index": OS_INDEX,
            "_id": doc["feed_id"],
            "_source": doc
        })

    success, failed = helpers.bulk(client, actions, stats_only=True)
    print(f"✅ Bulk push completed: {success} success, {failed} failed\n")
    print("-------------------------------------------------------------")
    
def test_opensearch_connection():
    try:
        # Ping the cluster
        if not client.ping():
            print("❌ Failed to connect to OpenSearch cluster.")
            return

        # Get basic info
        info = client.info()
        print("✅ Connected to OpenSearch")
        print(f"🧠 Cluster: {info['cluster_name']} | Version: {info['version']['number']}")

        # Optional: Check if the target index exists
        if client.indices.exists(index=OS_INDEX):
            print(f"📦 Index '{OS_INDEX}' exists.")
        else:
            print(f"⚠️ Index '{OS_INDEX}' does not exist.")

    except Exception as e:
        print(f"🔥 Connection test failed: {e}")
