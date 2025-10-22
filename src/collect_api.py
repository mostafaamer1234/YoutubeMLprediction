# src/collect_api.py
import argparse, os, math, time
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm

def get_api():
    load_dotenv()
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY not set in .env")
    return build("youtube", "v3", developerKey=key)

def fetch_trending(youtube, region="US", limit=2000):
    items = []
    page_token = None
    fetched = 0
    pbar = tqdm(total=limit, desc="Trending API")
    while fetched < limit:
        req = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode=region,
            maxResults=min(50, limit - fetched),
            pageToken=page_token
        )
        resp = req.execute()
        items.extend(resp.get("items", []))
        fetched += len(resp.get("items", []))
        pbar.update(len(resp.get("items", [])))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)
    pbar.close()
    return items

def fetch_search(youtube, query="music", limit=1000, publishedAfter=None):
    video_ids = []
    page_token = None
    fetched = 0
    pbar = tqdm(total=limit, desc="Search API ids")
    while fetched < limit:
        req = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=min(50, limit - fetched),
            pageToken=page_token,
            publishedAfter=publishedAfter
        )
        resp = req.execute()
        vids = [it["id"]["videoId"] for it in resp.get("items", []) if it["id"].get("videoId")]
        video_ids.extend(vids)
        fetched += len(vids)
        pbar.update(len(vids))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)
    pbar.close()
    # hydrate details
    details = []
    for i in tqdm(range(0, len(video_ids), 50), desc="Hydrate video details"):
        chunk = video_ids[i:i+50]
        r = youtube.videos().list(part="snippet,contentDetails,statistics", id=",".join(chunk)).execute()
        details.extend(r.get("items", []))
        time.sleep(0.05)
    return details

def fetch_channel_stats(youtube, channel_ids):
    out = {}
    for i in tqdm(range(0, len(channel_ids), 50), desc="Channel stats"):
        chunk = channel_ids[i:i+50]
        r = youtube.channels().list(part="statistics", id=",".join(chunk)).execute()
        for it in r.get("items", []):
            out[it["id"]] = it.get("statistics", {})
        time.sleep(0.05)
    return out

def normalize_items(items):
    rows = []
    for it in items:
        vid = it["id"]
        sn = it.get("snippet", {})
        st = it.get("statistics", {})
        cd = it.get("contentDetails", {})
        rows.append({
            "video_id": vid,
            "title": sn.get("title"),
            "description": sn.get("description"),
            "tags": ",".join(sn.get("tags", [])) if sn.get("tags") else None,
            "category_id": sn.get("categoryId"),
            "publish_date": sn.get("publishedAt"),
            "channel_title": sn.get("channelTitle"),
            "channel_id": sn.get("channelId"),
            "viewCount": int(st.get("viewCount", 0)) if st.get("viewCount") else 0,
            "likeCount": int(st.get("likeCount", 0)) if st.get("likeCount") else 0,
            "commentCount": int(st.get("commentCount", 0)) if st.get("commentCount") else 0,
            "duration_iso": cd.get("duration")
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["trending","search"], required=True)
    ap.add_argument("--region", default="US")
    ap.add_argument("--query", default="music")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    yt = get_api()
    if args.source == "trending":
        items = fetch_trending(yt, region=args.region, limit=args.limit)
    else:
        items = fetch_search(yt, query=args.query, limit=args.limit)

    df = normalize_items(items)

    # Attach channel subscriber counts
    ch_ids = list({c for c in df["channel_id"].dropna().unique()})
    ch_stats = fetch_channel_stats(yt, ch_ids)
    df["channel_subscribers"] = df["channel_id"].map(lambda cid: int(ch_stats.get(cid, {}).get("subscriberCount", 0)) if cid in ch_stats else 0)

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
