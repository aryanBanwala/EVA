# videos_extractor.py
import tempfile
import requests
import os

def download_video(url: str) -> str:
    """
    Downloads a video from the given URL into ./temp/ folder as a .mp4 file.
    Returns the local file path.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    filename = os.path.basename(url)
    file_path = os.path.join(TEMP_DIR, filename)


    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path


def delete_video(path: str) -> None:
    """
    Deletes the video file at the provided path if it exists.
    """
    try:
        os.remove(path)
    except OSError as e:
        print(f"⚠️ Could not delete {path}: {e}")

# Usage example:
# from videos_extractor import download_video, delete_video
#
# url = "https://example.com/video.mp4"
# path = download_video(url)
# # ... proceed with processing on `path` ...
# delete_video(path)