# backlog.py
"""
1) Pull a page of video-feed docs from Mongo, each with url=REL_PATH.
2) Download → frame-extract → embed in GPU/CPU batches.
3) Fill `vector` field in-memory AND push back to:
      • Qdrant (buffered, then flush once at the end)
      • Mongo (update_one per feed)                ← adjust if you have a helper
Environment variables (same keys you already use):
- PAGE, LIMIT, BATCH_SIZE, DEVICE, FRAME_PER_SECOND, FRAME_LIMIT,
  BASE_VIDEO_ENDPOINT, TIMEOUT_LIMIT, WANT_MEMORY_LOGS
"""

from datetime import datetime, timezone
from dotenv import load_dotenv
import os, sys, json, time, gc, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import torch

# ── Your internal libs ────────────────────────────────────────────────────────
from embeddings.video_embed import extract_and_preprocess_frames, embed_batch
from db.mongo.connection import init_connection
from db.mongo.feeds import (
    init as init_feeds,
    fetch_video_feeds_to_embed_backlog,
    update_vector_model_status
)
from utils.videos_extractor import download_video, delete_video
from db.opensearch.vector import bulk_push_vector_docs
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv(override=True)

# ── env ------------------------------------------------------------------
page          = int(os.getenv("PAGE", "1"))
limit         = int(os.getenv("LIMIT", "50"))
batch_size    = int(os.getenv("BATCH_SIZE", "5"))
device        = os.getenv("DEVICE", "cuda")
fps           = int(os.getenv("FRAME_PER_SECOND", "1"))
max_frames    = int(os.getenv("FRAME_LIMIT", "32"))
base_url      = os.getenv("BASE_VIDEO_ENDPOINT", "")
timeout_limit = int(os.getenv("TIMEOUT_LIMIT", "45"))
watch_logs    = os.getenv("WANT_MEMORY_LOGS", "false").lower() == "true"


# ============  Small utilities  ==============================================
def log_gpu_mem(tag=""):
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n📊 [GPU MEM] {tag}")
    print(f"  Allocated     : {torch.cuda.memory_allocated()   / 1024**2:.2f} MB")
    print(f"  Reserved      : {torch.cuda.memory_reserved()    / 1024**2:.2f} MB")
    print(f"  Max Allocated : {torch.cuda.max_memory_allocated()/ 1024**2:.2f} MB")
    print(f"  Max Reserved  : {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB\n")

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def download_and_extract(suffix_url, base_url, fps, max_frames, device):
    """
    Worker fn executed in ThreadPool threads: blocking I/O + CPU decode.
    Returns (rel_path, full_url, frames_tensor) or None on failure.
    """
    url = base_url+suffix_url
    try:
        tmp = download_video(url)
        frames = extract_and_preprocess_frames(tmp, fps, max_frames, device)
        delete_video(tmp)
        return suffix_url, url, frames
    except Exception as e:
        print(f"💥 Failed: {suffix_url}\n   ↳ {type(e).__name__}: {e}")
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except AttributeError:
                pass
    return None

# ============  Async Mongo Functions  ============================================
async def update_status_in_mongo(success_ids, failure_ids):
    """
    Wrapper that uses your bulk status updater + vector nulling helper.
    """
    # Set status flags
    if success_ids:
        await update_vector_model_status(success_ids, 1)
    if failure_ids:
        await update_vector_model_status(failure_ids, 0)


async def fetch_backlog(page: int, limit: int):
    await init_connection()
    init_feeds()
    feeds = await fetch_video_feeds_to_embed_backlog(page, limit)
    for feed in feeds:
        feed["vector"] = []
    return feeds

# ============ Elastic Placeholder ==========================================
def make_feed_id(doc_id, created_at):
    date_str = datetime.fromtimestamp(created_at, timezone.utc).strftime("%Y-%m-%d")
    return f"{doc_id}:{date_str}"

async def update_vector_in_elastic(documents):
    print(f"\n📤 [Elastic Update Payload] — {len(documents)} docs")

    prepared_docs = []

    for doc in documents:
        feed_id = make_feed_id(doc["feedId"], doc["createdAt"])
        prepared = {
            "feed_id": feed_id,
            "fileurl": doc.get("url"),
            "embedding": doc["vector"]
        }

        prepared_docs.append(prepared)

    bulk_push_vector_docs(prepared_docs)

# ============  Main ==========================================================
async def main():
    # ── fetch backlog --------------------------------------------------------
    feeds = await fetch_backlog(page, limit)
    batches = list(chunk_list(feeds, batch_size))
    total_batches = len(batches)
    print(f"🔄 Prepared {len(feeds)} feeds into {total_batches} batches (batch_size={batch_size})\n")

    success_ids = []
    failure_ids = []
    elastic_docs = []

    # ── main batch loop ------------------------------------------------------
    for batch_idx, batch in enumerate(batches, start=1):
        print(f"—— Batch {batch_idx}/{total_batches} ({len(batch)} videos) ——")

        exe = ThreadPoolExecutor(max_workers=len(batch))
        futures = {}
        for i, feed in enumerate(batch):
            futures[exe.submit(download_and_extract,
                               feed["url"], base_url,
                               fps, max_frames, device)] = i

        results = []
        failed_downloads = set()

        try:
            for fut in as_completed(futures, timeout=timeout_limit):
                i = futures[fut]
                feed_doc = batch[i]
                try:
                    res = fut.result()
                    if res is not None:
                        results.append((i, res))
                    else:
                        failed_downloads.add(i)
                        print(f"❌ Download failed: {feed_doc['url']}")
                except Exception as e:
                    failed_downloads.add(i)
                    print(f"💥 Worker error for {feed_doc['url']}: {e}")
        except TimeoutError:
            print(f"⏱️ Batch-timeout ({timeout_limit}s) — continuing with partial results")
        finally:
            exe.shutdown(wait=False, cancel_futures=True)

        # reorder and extract just the tensors
        results.sort(key=lambda t: t[0])
        frames_list = [r[1][2] for r in results]

        # embedding pass (only on downloaded frames)
        embeddings = []
        try:
            embeddings = embed_batch(frames_list, device)
        except Exception as e:
            print(f"💥 Embedding error: {e}")

        # assign vectors & collect successes
        for (order_idx, (rel_path, url, _)), emb in zip(results, embeddings):
            feed_doc = batch[order_idx]
            vector = emb.tolist()
            feed_doc["vector"] = vector
            print(f"{feed_doc['feedId']}  {vector[:3]}  {len(vector)}")
            success_ids.append(feed_doc["feedId"])
            elastic_docs.append({
                "feedId": feed_doc["feedId"],
                "url": feed_doc.get("url"),
                "createdAt": feed_doc.get("created_at"),
                "vector": vector
            })

        # anything not in success_ids from this batch is a failure
        for feed in batch:
            if feed["feedId"] not in success_ids:
                failure_ids.append(feed["feedId"])

        # clean up GPU
        if device == "cuda":
            del results, frames_list, embeddings
            gc.collect()
            torch.cuda.empty_cache()
            try: torch.cuda.ipc_collect()
            except AttributeError: pass
            if watch_logs and batch_idx % 10 == 0:
                log_gpu_mem(f"after batch {batch_idx}")

    await update_status_in_mongo(success_ids, failure_ids)
    await update_vector_in_elastic(elastic_docs)

# ============  Entry-point ====================================================
async def main_process():
    while True:
        try:
            await main()
        except Exception as e:
            time.sleep(5)


if __name__ == "__main__":
    asyncio.run(main_process())