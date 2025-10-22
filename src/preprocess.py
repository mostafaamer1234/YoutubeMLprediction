# src/preprocess.py
import argparse, re, math, pandas as pd, numpy as np
from datetime import datetime, timezone
import regex as regex

ISO_PATTERN = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")

def parse_iso_duration(s):
    if not isinstance(s, str):
        return np.nan
    m = ISO_PATTERN.match(s)
    if not m:
        return np.nan
    h = int(m.group(1) or 0)
    mnt = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return h*3600 + mnt*60 + sec

def parse_views_text(s):
    if not isinstance(s, str): return np.nan
    s = s.lower().replace(",", "")
    m = re.search(r"([\d\.]+)\s*([km]?)\s*views?", s)
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        if suf == "k": val *= 1e3
        elif suf == "m": val *= 1e6
        return int(val)
    # fall back to digits
    digits = re.findall(r"\d+", s.replace(",", ""))
    if digits:
        try:
            return int("".join(digits))
        except: return np.nan
    return np.nan

def to_days_since(published_at):
    if not isinstance(published_at, str): return np.nan
    try:
        dt = datetime.fromisoformat(published_at.replace("Z","+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return np.nan

def basic_title_feats(title):
    if not isinstance(title, str): return {"title_len":0,"title_caps_ratio":0.0,"has_numbers_title":0}
    length = len(title)
    caps = sum(1 for c in title if c.isupper())
    caps_ratio = caps / max(length,1)
    has_num = int(any(ch.isdigit() for ch in title))
    return {"title_len": length, "title_caps_ratio": caps_ratio, "has_numbers_title": has_num}

def keyword_counts(title, vocab):
    if not isinstance(title, str): return {f"kw_{w}":0 for w in vocab}
    low = title.lower()
    return {f"kw_{w}": int(w in low) for w in vocab}

def process_scraped(df):
    out = df.copy()
    out["views"] = df["views_text"].apply(parse_views_text).fillna(0).astype(int)
    # duration may be "12:34"
    def dur_to_sec(s):
        if not isinstance(s,str): return np.nan
        parts = s.strip().split(":")
        if len(parts)==2:
            m,s = parts
            return int(m)*60+int(s)
        if len(parts)==3:
            h,m,s = parts
            return int(h)*3600+int(m)*60+int(s)
        return np.nan
    out["duration_seconds"] = df["duration_text"].apply(dur_to_sec)
    out["publish_age_days"] = np.nan  # unknown in scrape
    # Title features
    feats = df["title"].apply(basic_title_feats).apply(pd.Series)
    out = pd.concat([out, feats], axis=1)
    # Simple vocab
    vocab = ["official","trailer","live","remix","tutorial","news","review","shorts","asmr","vlog"]
    kws = df["title"].apply(lambda t: keyword_counts(t, vocab)).apply(pd.Series)
    out = pd.concat([out, kws], axis=1)
    # engagement placeholders (unknown likes/comments in scrape)
    out["likes"] = np.nan
    out["comments"] = np.nan
    out["engagement_rate"] = np.nan
    # category_from_title guess (very naive)
    def guess_cat(title):
        t = (title or "").lower()
        if any(w in t for w in ["official video","remix","lyrics","feat"]): return "Music"
        if "trailer" in t: return "Film & Animation"
        if any(w in t for w in ["review","unboxing","vs "]): return "Tech/Reviews"
        if "news" in t: return "News & Politics"
        if any(w in t for w in ["vlog","day in the life"]): return "People & Blogs"
        return None
    out["category_guess"] = df["title"].apply(guess_cat)
    return out

def process_api(df):
    out = df.copy()
    out["duration_seconds"] = df["duration_iso"].apply(parse_iso_duration)
    out["publish_age_days"] = df["publish_date"].apply(to_days_since)
    out["title_len"] = df["title"].apply(lambda t: basic_title_feats(t)["title_len"])
    out["title_caps_ratio"] = df["title"].apply(lambda t: basic_title_feats(t)["title_caps_ratio"])
    out["has_numbers_title"] = df["title"].apply(lambda t: basic_title_feats(t)["has_numbers_title"])
    vocab = ["official","trailer","live","remix","tutorial","news","review","shorts","asmr","vlog"]
    for w in vocab:
        out[f"kw_{w}"] = df["title"].str.lower().str.contains(w, na=False).astype(int)
    # views/likes/comments already numeric
    out["views"] = df["viewCount"].fillna(0).astype(int)
    out["likes"] = df["likeCount"].fillna(0).astype(int)
    out["comments"] = df["commentCount"].fillna(0).astype(int)
    out["engagement_rate"] = (out["likes"] + out["comments"]) / out["views"].replace(0, np.nan)
    out["engagement_rate"] = out["engagement_rate"].fillna(0.0)
    # category title mapping is optional; leave category_id as proxy
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scraped", nargs="*", default=[])
    ap.add_argument("--api", nargs="*", default=[])
    ap.add_argument("--out_scraped", required=True)
    ap.add_argument("--out_api", required=True)
    args = ap.parse_args()

    # Load
    import os
    scraped_frames = []
    for p in args.scraped:
        try:
            scraped_frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
    api_frames = []
    for p in args.api:
        try:
            api_frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")

    scraped = pd.concat(scraped_frames, ignore_index=True) if scraped_frames else pd.DataFrame()
    api = pd.concat(api_frames, ignore_index=True) if api_frames else pd.DataFrame()

    if not scraped.empty:
        scraped_clean = process_scraped(scraped)
        scraped_clean.to_csv(args.out_scraped, index=False)
        print(f"Saved scraped preprocessed to {args.out_scraped} ({len(scraped_clean)})")
    else:
        print("No scraped data provided")

    if not api.empty:
        api_clean = process_api(api)
        api_clean.to_csv(args.out_api, index=False)
        print(f"Saved api preprocessed to {args.out_api} ({len(api_clean)})")
    else:
        print("No api data provided")

if __name__ == "__main__":
    main()
