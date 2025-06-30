import os
import sys
import json
import time
import shutil
from dotenv import load_dotenv
from embeddings.video_embed import get_video_embedding
from db.qdrant              import buffer_point, flush_buffer
from utils.videos_extractor import download_video, delete_video


load_dotenv(override=True)
try:
    num = os.environ["NUM"]
    device = os.environ["DEVICE"]
    fps = os.environ["FRAME_PER_SECOND"]
    frame_limit = os.environ["FRAME_LIMIT"]
    BASE_VIDEO_ENPOINT = os.environ["BASE_VIDEO_ENPOINT"]
except KeyError as e:
    print(f"❌ Missing environment variable: {e}")
    sys.exit(1)

def process_file(
    video_path,
    collection_name,
    frame_per_second=1,
    max_frames=30,
    device='cpu',
    url="default"
):
    embedding = get_video_embedding(
        video_path=video_path,
        frame_per_second=frame_per_second,
        max_frames=max_frames,
        device=device
    )
    buffer_point(
        collection_name=collection_name,
        vector=embedding.tolist(),
        payload={"fileurl": url}
    )
    

def process_from_json(
    json_path,
    base_url,
    collection_name,
    frame_per_second=1,
    max_frames=30,
    device='cpu'
):
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        rel_paths = json.load(f)

    total = len(rel_paths)
    print(f"📦 Total videos to process: {total}\n")

    for i, rel in enumerate(rel_paths, start=1):
        full_url = f"{base_url.rstrip('/')}/{rel.lstrip('/')}"
        tmp_path = None
        print(f"🔄 [{i}/{total}] Downloading → {rel}")

        try:
            tmp_path = download_video(full_url)
            print(f"📥 [{i}/{total}] Downloaded")

            print(f"🧠 [{i}/{total}] Generating embedding")
            process_file(
                video_path=tmp_path,
                collection_name=collection_name,
                frame_per_second=frame_per_second,
                max_frames=max_frames,
                device=device,
                url=full_url
            )
            print(f"✅ [{i}/{total}] Done: {rel}\n")

        except Exception as e:
            print(f"❌ [{i}/{total}] Failed: {rel}\n   → {e}\n")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                delete_video(tmp_path)

if __name__ == "__main__":
    start = time.time()
    print("🚀 Job started\n")
    json_name = "feeds_clips_" + num

    process_from_json(
        json_path="assets/" + json_name + ".json",
        base_url=BASE_VIDEO_ENPOINT,
        collection_name=json_name,
        frame_per_second=int(fps),
        max_frames=int(frame_limit),
        device=device
    )
    
    total_time = time.time() - start
    mins, secs = divmod(total_time, 60)
    print(f"\n🎬 Video processing complete in {int(mins)} min {int(secs)} sec.")

    # Start timing bulk upload
    upload_start = time.time()
    print(f"\n📤 Starting bulk upload to Qdrant for collection: '{json_name}'")
    flush_buffer(json_name)
    upload_time = time.time() - upload_start
    upload_mins, upload_secs = divmod(upload_time, 60)
    print(f"✅ Bulk upload complete in {int(upload_mins)} min {int(upload_secs)} sec.")

    # Grand total
    grand_total = time.time() - start
    grand_mins, grand_secs = divmod(grand_total, 60)
    print(f"\n🏁 Total job finished in {int(grand_mins)} min {int(grand_secs)} sec.\n")