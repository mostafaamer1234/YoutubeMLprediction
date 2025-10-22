# src/scrape_youtube.py
import argparse, time, re, csv, sys
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd

def extract_video_id(url: str):
    try:
        q = parse_qs(urlparse(url).query)
        return q.get("v", [None])[0]
    except Exception:
        return None

def scroll_to_load(driver, limit=1000, pause=1.0):
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    items_seen = 0
    while items_seen < limit:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        items_seen += 50

def parse_cards(html):
    soup = BeautifulSoup(html, "html.parser")
    data = []
    for a in soup.select("a#video-title"):
        title = a.get("title") or a.text.strip()
        url = "https://www.youtube.com" + a.get("href", "")
        vid = extract_video_id(url)
        # Attempt to get container for channel & metadata
        parent = a.find_parent("ytd-video-renderer") or a.find_parent("ytd-grid-video-renderer") or a.find_parent("ytd-video-renderer")
        channel = None
        views_text = None
        time_text = None
        duration_text = None
        if parent:
            ch = parent.select_one("a.yt-simple-endpoint.style-scope.yt-formatted-string")
            channel = ch.text.strip() if ch else None
            meta = parent.select("span.inline-metadata-item")
            if meta and len(meta) >= 2:
                views_text = meta[0].text.strip()
                time_text = meta[1].text.strip()
            dur = parent.select_one("span.ytd-thumbnail-overlay-time-status-renderer")
            duration_text = dur.text.strip() if dur else None
        data.append({
            "title": title, "url": url, "video_id": vid,
            "channel_name": channel,
            "views_text": views_text, "time_text": time_text,
            "duration_text": duration_text
        })
    # De-duplicate by video_id/url
    seen = set()
    out = []
    for d in data:
        key = d.get("video_id") or d.get("url")
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trending","search"], required=True)
    ap.add_argument("--query", default=None, help="Search query text for mode=search")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--lang=en-US")
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=options)

    try:
        if args.mode == "trending":
            url = "https://www.youtube.com/feed/trending"
        else:
            q = args.query or "music"
            url = f"https://www.youtube.com/results?search_query={q}&sp=CAI%253D"  
        driver.get(url)
        time.sleep(3)
        scroll_to_load(driver, limit=args.limit, pause=1.0)
        html = driver.page_source
        rows = parse_cards(html)
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} rows to {args.out}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
