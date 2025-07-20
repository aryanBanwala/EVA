# migration/qdrant_to_opensearch.py
import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from db.opensearch import (
    client as os_client,
    ensure_index,
    buffer_point,
    flush_buffer,
    BUFFER
)

load_dotenv(override=True)


def human(n):
    return f"{n:,}"


def verify_connections():
    # Qdrant configuration
    qurl = os.getenv("QDRANT_URL")
    qkey = os.getenv("QDRANT_API_KEY")
    coll = os.getenv("QDRANT_COLLECTION") or f"feeds_clips_{os.getenv('NUM')}"
    # OpenSearch index name
    index = os.getenv("OS_INDEX") or os.getenv("ES_INDEX")

    print(f"\n🔍  QDRANT_URL={qurl}\n     COLLECTION={coll}\n     OS_INDEX={index}\n")
    if not all([qurl, qkey, coll, index]):
        print("❌  Missing environment variables. Please set QDRANT_URL, QDRANT_API_KEY, OS_INDEX.")
        return None

    # Verify Qdrant connectivity
    try:
        qc = QdrantClient(url=qurl, api_key=qkey)
        info = qc.get_collection(coll)
        total = info.points_count
        print(f"✅  Qdrant OK → {human(total)} vectors\n")
    except Exception as e:
        print("❌  Qdrant error:", e)
        return None

    # Verify OpenSearch connectivity
    if not os_client.ping():
        print("❌  OpenSearch ping failed")
        return None
    print("✅  OpenSearch OK\n")

    return {
        "qclient": qc,
        "collection": coll,
        "index": index,
        "total": total
    }


def migrate_qdrant_to_opensearch(batch_size=500, test_phase=True):
    cfg = verify_connections()
    if not cfg:
        return
    # Clear any existing buffer
    BUFFER.clear()

    qc = cfg["qclient"]
    coll = cfg["collection"]
    index = cfg["index"]
    total_expected = cfg["total"]

    # Fetch one sample to detect vector dimension
    sample, _ = qc.scroll(coll, limit=1, with_vectors=True, with_payload=True)
    dim = len(sample[0].vector)
    # Ensure index exists in OpenSearch
    ensure_index(index, dim)

    print(f"🚚  Starting migration: batch_size={batch_size}, expect {human(total_expected)} vectors\n")

    offset = None
    total = 0
    batch_no = 1
    start_all = time.time()

    while total < total_expected:
        start_batch = time.time()
        pts, offset = qc.scroll(
            coll,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        if not pts:
            break

        # Buffer points
        for p in pts:
            buffer_point(index, p.vector, p.payload or {})

        if test_phase:
            print(f"🧪  Test mode → buffered {len(pts)} vectors; exiting before flush.")
            break

        # Flush to OpenSearch
        flush_buffer(index)

        sent = len(pts)
        total += sent
        duration = time.time() - start_batch
        print(f"📦  Batch {batch_no:<3}| {human(sent):>5} vec | {duration:>5.2f}s | total {human(total)}")
        batch_no += 1

    elapsed = time.time() - start_all
    if not test_phase:
        print(f"\n✅  Migration complete: {human(total)} / {human(total_expected)} vectors in {elapsed:.2f}s")


if __name__ == "__main__":
    # Default to test_phase=True unless explicitly set to 'false'
    test_flag = os.getenv("TEST_PHASE", "true").lower() != "false"
    batch = int(os.getenv("BATCH_SIZE", "500"))
    migrate_qdrant_to_opensearch(batch_size=batch, test_phase=test_flag)