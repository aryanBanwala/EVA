import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from collections import Counter
from opensearchpy import OpenSearch


# ——— Static config ———
MONGO_URI       = "mongodb+srv://aryanRO:htPdaHYGX8Fto6DM@chatwise-production.eyezap.mongodb.net/"
DB_NAME         = "jigrrapi"
COLLECTION_NAME = "feeds"
FILE_LIST_PATH  = "assets/file_urls_for_mapping.json"
OUTPUT_PATH     = "assets/feed_url_id_map.json"
LOG_EVERY       = 100

ES_HOST              = os.getenv("OS_HOST", "localhost")
ES_PORT              = int(os.getenv("OS_PORT", "9200"))
ES_INDEX             = os.getenv("OS_INDEX", "test-vectors")
BASE_VIDEO_ENDPOINT  = os.getenv("BASE_VIDEO_ENPOINT",
                             "https://india-media-downlink-preprod.chatwise.co.uk/videos/crop/")

# Init ES client
es = OpenSearch(
    hosts=[{"host": ES_HOST, "port": ES_PORT}],
    use_ssl=False
)

def make_feed_id(doc_id, created_at):
    date_str = datetime.utcfromtimestamp(created_at).strftime("%Y-%m-%d")
    return f"{doc_id}:{date_str}"

def extract_media_url(media_arr):
    """
    Return first element's 'url' if present, else None.
    """
    if not media_arr:
        return None
    first = media_arr[0]
    return first.get("url")

def main():
    # 1) Load stripped URLs
    with open(FILE_LIST_PATH, "r") as f:
        stripped_urls = json.load(f)

    total = len(stripped_urls)
    print(f"📥 Loaded {total} URLs")

    # 2) Mongo connection
    client = MongoClient(MONGO_URI)
    col    = client[DB_NAME][COLLECTION_NAME]

    # 3) Bulk fetch
    cursor = col.find(
        {"data.media.0.url": {"$in": stripped_urls}},
        {"_id": 1, "created_at": 1, "data.media": 1}   # grab full media[0] object
    )

    lookup = {}
    for doc in cursor:
        url = extract_media_url(doc.get("data", {}).get("media"))
        if not url:
            continue
        lookup[url] = (str(doc["_id"]), doc.get("created_at"))

    # 4) Process in original order
    mapped, with_ct, without_ct = [], 0, 0
    for i, url in enumerate(stripped_urls, 1):
        entry = {"url": url}
        hit   = lookup.get(url)

        if hit and hit[1] is not None:
            _id, ct = hit
            entry.update({
                "_id": _id,
                "created_at": ct,
                "feed_id": make_feed_id(_id, ct)
            })
            with_ct += 1
        else:
            entry.update({"_id": None, "created_at": None, "feed_id": None})
            without_ct += 1

        mapped.append(entry)
        if i % LOG_EVERY == 0 or i == total:
            print(f"🔄 Processed {i}/{total}")

    # 5) Save
    with open(OUTPUT_PATH, "w") as w:
        json.dump(mapped, w, indent=2)

    # 6) Summary
    print("\n📊 Summary")
    print(f"➡️ Total URLs:             {total}")
    print(f"✅ With created_at:        {with_ct}")
    print(f"❌ Missing created_at:     {without_ct}")
    print(f"💾 Output saved to: {OUTPUT_PATH}")


def validate_mapping(path):
    # load mapping JSON
    with open(path, "r") as f:
        data = json.load(f)

    total = len(data)
    print(f"📄 Loaded {total} records for validation\n")

    # track presence + uniqueness
    ids, ats, fids, urls = set(), set(), set(), set()
    missing_keys = 0
    non_unique = {"_id":0, "created_at":0, "feed_id":0, "url":0}

    for entry in data:
        _id = entry.get("_id")
        at  = entry.get("created_at")
        fid = entry.get("feed_id")
        url = entry.get("url")

        # presence check
        if None in (_id, at, fid, url):
            missing_keys += 1
            continue

        # uniqueness checks
        if _id in ids:      non_unique["_id"]       += 1
        if at  in ats:      non_unique["created_at"] += 1
        if fid in fids:     non_unique["feed_id"]    += 1
        if url in urls:     non_unique["url"]        += 1

        ids.add(_id); ats.add(at); fids.add(fid); urls.add(url)

    # basic summary
    print("📊 Validation Results")
    print(f"➡️ Total entries:             {total}")
    print(f"✅ Complete entries:          {total - missing_keys}")
    print(f"❌ Missing fields:            {missing_keys}\n")

    print("🔐 Uniqueness check:")
    print(f" - Unique _id:        {len(ids)}")
    print(f" - Unique created_at: {len(ats)}")
    print(f" - Unique feed_id:    {len(fids)}")
    print(f" - Unique url:        {len(urls)}\n")

    print("⚠️  Non-unique counts:")
    for k, v in non_unique.items():
        print(f" - {k}: {v}")
    print()

    # find duplicate timestamps
    all_ats = [e["created_at"] for e in data if e.get("created_at") is not None]
    dup_counts = {ts:cnt for ts, cnt in Counter(all_ats).items() if cnt > 1}
    print(f"🔍 Found {len(dup_counts)} duplicate created_at values\n")

    # connect Mongo
    client = MongoClient(MONGO_URI)
    col    = client[DB_NAME][COLLECTION_NAME]

    # for each duplicate ts, query and log docs
    for idx, (ts, cnt) in enumerate(dup_counts.items(), start=1):
        print(f"=== [{idx}/{len(dup_counts)}] created_at = {ts} (count={cnt}) ===")
        cursor = col.find(
            {"created_at": ts},
            {"_id":1, "created_at":1, "data.media":1}
        )
        for doc in cursor:
            media = doc.get("data",{}).get("media",[])
            url   = media[0].get("url") if media else None
            fid   = make_feed_id(str(doc["_id"]), ts)
            print(f"  • _id: {doc['_id']}, url: {url}, feed_id: {fid}")
        print()
        
def update_es_with_feed_id(mapping_path):
    """
    For each entry in the mapping JSON, add a `feed_id` field
    to the ES documents whose `fileURL` matches the full URL.
    """
    with open(mapping_path, "r") as f:
        data = json.load(f)

    total = len(data)
    print(f"\n🔧 Updating ES index '{ES_INDEX}' with feed_id (total {total})\n")

    for i, entry in enumerate(data, start=1):
        stripped = entry.get("url")
        feed_id  = entry.get("feed_id")
        if not stripped or not feed_id:
            continue

        # Build the full URL exactly as it's in ES
        full_url = BASE_VIDEO_ENDPOINT + stripped

        # Update-by-query: set ctx._source.feed_id
        body = {
            "script": {
                "source": "ctx._source.feed_id = params.feed_id",
                "lang":   "painless",
                "params": {"feed_id": feed_id}
            },
            "query": {
                "term": {
                    "fileurl.keyword": full_url
                }
            }
        }

        resp = es.update_by_query(
            index       = ES_INDEX,
            body        = body,
            refresh     = True,       # make change visible immediately
            conflicts   = "proceed"   # skip any version conflicts
        )

        updated = resp.get("updated", 0)
        print(f"🔄 [{i}/{total}] `{stripped}` → updated {updated} doc(s)")

    print("\n✅ Done pushing feed_id into ES.")

def verify_es_data(mapping_path, chunk_size=500):
    """
    1️⃣  Load the mapping JSON.  
    2️⃣  Fetch docs from OpenSearch in bulk (chunked *terms* query on `fileurl.keyword`).  
    3️⃣  Check that each doc exists **and** carries the expected `feed_id`.  
    4️⃣  Print a concise summary with mismatches & misses.

    ‼️  Requires the global `es`, `BASE_VIDEO_ENDPOINT`, and `ES_INDEX`
        already defined above.
    """
    # -- Load mapping -----------------------------------------------------
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    total           = len(mapping)
    expected_urls   = [BASE_VIDEO_ENDPOINT + e["url"] for e in mapping]
    expected_lookup = {BASE_VIDEO_ENDPOINT + e["url"]: e["feed_id"] for e in mapping}

    print(f"🔍 Verifying {total} records against index '{ES_INDEX}'")

    # -- Stats holders ----------------------------------------------------
    found_doc          = 0
    missing_doc        = 0
    feed_id_match      = 0
    feed_id_mismatch   = 0
    mismatched_details = []       # (url, expected, actual)

    # -- Chunked ES fetch -------------------------------------------------
    for i in range(0, total, chunk_size):
        chunk_urls = expected_urls[i:i + chunk_size]

        body = {
            "size": len(chunk_urls),
            "_source": ["fileurl", "feed_id"],
            "query": {
                "terms": {
                    "fileurl.keyword": chunk_urls
                }
            }
        }

        res   = es.search(index=ES_INDEX, body=body)
        hits  = res.get("hits", {}).get("hits", [])

        # Map returned docs by url for quick lookup
        hit_map = {
            h["_source"]["fileurl"]: h["_source"].get("feed_id")
            for h in hits
        }

        # -- Evaluate this chunk -----------------------------------------
        for url in chunk_urls:
            expected_feed_id = expected_lookup[url]
            actual_feed_id   = hit_map.get(url)

            if actual_feed_id is None:
                missing_doc += 1
                continue

            found_doc += 1
            if actual_feed_id == expected_feed_id:
                feed_id_match += 1
            else:
                feed_id_mismatch += 1
                mismatched_details.append((url, expected_feed_id, actual_feed_id))

        print(f"  • Processed {min(i + chunk_size, total)}/{total}")

    # -- Summary ----------------------------------------------------------
    print("\n📊  Verification Summary")
    print(f"   ↪ Total entries checked : {total}")
    print(f"   ✅ Docs found           : {found_doc}")
    print(f"   ❌ Docs missing         : {missing_doc}")
    print(f"   🔒 feed_id OK           : {feed_id_match}")
    print(f"   ⚠️  feed_id mismatched  : {feed_id_mismatch}")

    # List mismatches (if any)
    if mismatched_details:
        print("\n  MISMATCHED feed_id DETAILS")
        for url, exp, act in mismatched_details:
            print(f"   - {url}\n       expected: {exp}\n       actual  : {act}")

verify_es_data(OUTPUT_PATH)