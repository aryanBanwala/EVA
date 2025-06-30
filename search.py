import os
import sys
from dotenv import load_dotenv
from embeddings.text_embed import get_text_embedding
from db.qdrant import search_similar_vectors


load_dotenv(override=True)
try:
    k = int(os.environ["K"])
    num = os.environ["NUM"]
    device = os.environ["DEVICE"]
except KeyError as e:
    print(f"❌ Missing environment variable: {e}")
    sys.exit(1)
except ValueError as e:
    print(f"❌ Invalid value in env: {e}")
    sys.exit(1)


def run_text_search(query, collection_name="feeds_clips_1", top_k=5, device="cpu"):
    print(f"🔎 Searching for: '{query}'")

    # 1) Embed the query in CLIP space
    q_vec = get_text_embedding(query, device)

    # 2) Ask Qdrant for the nearest video vectors
    hits = search_similar_vectors(collection_name, q_vec, top_k)

    # 3) Display results
    print(f"\n🎯 Top {top_k} Results:")
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        print(
            f"{i}. ID: {hit.id} | Score: {hit.score:.4f}\n"
            f"   File: {payload.get('fileurl', 'N/A')}"
        )


if __name__ == "__main__":
    query = input("Enter your search query: ")
    
    while(query != "end"):
        run_text_search(
                query = query,
                collection_name="feeds_clips_"+num,
                top_k=k,
                device=device
                )
        
        query = input("Enter your search query: ")