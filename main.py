import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import json
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed , TimeoutError

from embeddings.video_embed import extract_and_preprocess_frames, embed_batch
from db.qdrant import buffer_point, flush_buffer
from utils.videos_extractor import download_video, delete_video

import torch
import gc

def log_gpu_mem(tag=""):
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n📊 [GPU MEM] {tag}")
    print(f"  Allocated     : {torch.cuda.memory_allocated()   / 1024**2:.2f} MB")
    print(f"  Reserved      : {torch.cuda.memory_reserved()    / 1024**2:.2f} MB")
    print(f"  Max Allocated : {torch.cuda.max_memory_allocated()/ 1024**2:.2f} MB")
    print(f"  Max Reserved  : {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB\n")


def download_and_extract(rel_path, base_url, fps, max_frames, device):
    url = f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}"

    try:
        tmp = download_video(url)
        frames = extract_and_preprocess_frames(tmp, fps, max_frames, device)
        delete_video(tmp)
        return rel_path, url, frames

    except Exception as e:
        print(f"💥 Failed: {rel_path}\n   ↳ {type(e).__name__}: {e}")

    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except AttributeError:
                pass

    return None


def main():
    load_dotenv(override=True)
    try:
        num         = os.environ["NUM"]
        device      = os.environ["DEVICE"]
        fps         = int(os.environ["FRAME_PER_SECOND"])
        max_frames  = int(os.environ["FRAME_LIMIT"])
        base_url    = os.environ["BASE_VIDEO_ENPOINT"]
        batch_size = int(os.environ["BATCH_SIZE"])
        watch_logs = os.environ["WANT_MEMORY_LOGS"].lower() == "true"
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        sys.exit(1)

    collection = f"feeds_clips_{num}"
    json_path  = f"assets/{collection}.json"

    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        rel_paths = json.load(f)

    total = len(rel_paths)
    start = time.time()
    total_batches = (total + batch_size - 1) // batch_size
    for batch_start in range(0, total, batch_size):
        batch_num = batch_start // batch_size + 1
        batch = rel_paths[batch_start: batch_start + batch_size]
        print(f"🔄 Batch [{batch_num}/{total_batches}]: {len(batch)} videos")

        # download & preprocess in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(batch))) as exe:
            futures = {
                exe.submit(download_and_extract, rel, base_url, fps, max_frames, device): rel
                for rel in batch
            }

            results = []
            try:
                # wait up to 45s for *all* futures
                for fut in as_completed(futures, timeout=45):
                    rel = futures[fut]
                    try:
                        res = fut.result()
                        if res is not None:
                            results.append(res)
                    except Exception as e:
                        print(f"💥 Skipped {rel}: {type(e).__name__}: {e}")

            except TimeoutError:
                print(f"⏱️ Batch-timeout (45 s) — Skipping entire batch {batch_num}/{total_batches}")
                for fut in futures:
                    if not fut.done():
                        fut.cancel()
                continue  # ← go straight to the next batch

        # if we get here, results contains *all* successful video extractions
        if not results:
            print(f"⚠️  Batch [{batch_num}/{total_batches}] skipped — all videos failed.")
            continue
                   
        # reorder to original batch order
        results.sort(key=lambda x: batch.index(x[0]))
        frames_list = [r[2] for r in results]

        # one-shot embed
        embeddings = embed_batch(frames_list, device)
        
        # buffer to Qdrant
        for (rel, url, _), emb in zip(results, embeddings):
            # print(f"✅ Embedded & buffering: {rel}")
            buffer_point(collection, vector=emb.tolist(), payload={"fileurl": url})

        # ── FREE FRAME TENSORS & FORCE GC ─────────────────────────────────
        if device == "cuda":
            for _, _, frames in results:
                del frames
            # Drop references to these collections
            del results, frames_list, embeddings
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except AttributeError:
                pass
            if(watch_logs and batch_num%10 == 0):
                log_gpu_mem()

    print(f"\n📤 Flushing buffer to Qdrant: '{collection}'")
    flush_buffer(collection)

    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    print(f"\n🏁 All done in {int(m)}m {int(s)}s")

if __name__ == "__main__":
    main()