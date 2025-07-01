import os
import sys
import json
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from embeddings.video_embed import extract_and_preprocess_frames, embed_batch
from db.qdrant import buffer_point, flush_buffer
from utils.videos_extractor import download_video, delete_video

def download_and_extract(rel_path, base_url, fps, max_frames, device):
    url = f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}"
    tmp = download_video(url)
    frames = extract_and_preprocess_frames(tmp, fps, max_frames, device)
    delete_video(tmp)
    return rel_path, url, frames

def main():
    load_dotenv(override=True)
    try:
        num         = os.environ["NUM"]
        device      = os.environ["DEVICE"]
        fps         = int(os.environ["FRAME_PER_SECOND"])
        max_frames  = int(os.environ["FRAME_LIMIT"])
        base_url    = os.environ["BASE_VIDEO_ENPOINT"]
        batch_size = int(os.environ["BATCH_SIZE"])
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

    total      = len(rel_paths)
    start = time.time()
    for batch_start in range(0, total, batch_size):
        batch = rel_paths[batch_start: batch_start + batch_size]
        print(f"🔄 Batch {batch_start//batch_size+1}: {len(batch)} videos")

        # download & preprocess in parallel
        with ThreadPoolExecutor(max_workers=len(batch)) as exe:
            futures = [
                exe.submit(download_and_extract, rel, base_url, fps, max_frames, device)
                for rel in batch
            ]
            results = [f.result() for f in as_completed(futures)]

        # reorder to original batch order
        results.sort(key=lambda x: batch.index(x[0]))
        frames_list = [r[2] for r in results]

        # one-shot embed
        embeddings = embed_batch(frames_list, device)

        # buffer to Qdrant
        for (rel, url, _), emb in zip(results, embeddings):
            print(f"✅ Embedded & buffering: {rel}")
            buffer_point(collection, vector=emb.tolist(), payload={"fileurl": url})

    print(f"\n📤 Flushing buffer to Qdrant: '{collection}'")
    flush_buffer(collection)

    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    print(f"\n🏁 All done in {int(m)}m {int(s)}s")

if __name__ == "__main__":
    main()