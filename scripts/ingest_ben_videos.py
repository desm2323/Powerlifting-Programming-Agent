"""
ingest_ben_videos.py — fetch YouTube transcripts for the Ben Johnson
videos that ground the expert_knowledge.py KB.

This is a BUILD-TIME script. The agent itself never calls YouTube at
runtime — transcripts get fetched once and committed to
docs/ben_transcripts/, where they sit as static citation material.

Usage:
    python scripts/ingest_ben_videos.py docs/ben_videos.txt

Output:
    docs/ben_transcripts/<video_id>.txt   (one per video, with metadata header)

Dependencies:
    pip install youtube-transcript-api
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


YOUTUBE_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str | None:
    m = YOUTUBE_VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def fetch_metadata(url: str) -> dict:
    """Fetch title + author via YouTube's keyless oEmbed endpoint."""
    oembed_url = ("https://www.youtube.com/oembed"
                  f"?url={urllib.parse.quote(url, safe='')}&format=json")
    try:
        with urllib.request.urlopen(oembed_url, timeout=10) as r:
            return json.load(r)
    except Exception as e:
        return {"title": "(metadata unavailable)", "author_name": "",
                "error": str(e)}


def fetch_transcript(video_id: str):
    """Return a list of {text, start, duration} entries or None on failure.

    Handles both the legacy (0.x) classmethod API and the newer (1.x)
    instance API of youtube-transcript-api.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("ERROR: youtube-transcript-api not installed.\n"
              "Install with: pip install youtube-transcript-api",
              file=sys.stderr)
        sys.exit(1)

    languages = ["en", "en-US", "en-GB"]

    # Try legacy classmethod API first (0.6.x).
    get_transcript = getattr(YouTubeTranscriptApi, "get_transcript", None)
    if callable(get_transcript):
        try:
            return get_transcript(video_id, languages=languages)
        except Exception as e:
            print(f"  ! legacy fetch failed for {video_id}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)

    # Fall back to instance API (1.x).
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        # FetchedTranscript exposes .snippets; each snippet has .text / .start.
        snippets = getattr(fetched, "snippets", None) or list(fetched)
        out = []
        for s in snippets:
            # Newer lib uses object attrs; legacy uses dicts. Don't mix the
            # two with `or` chains — an empty .text trips the dict path.
            if hasattr(s, "text"):
                text, start, duration = s.text, s.start, s.duration
            else:
                text = s.get("text", "")
                start = s.get("start", 0.0)
                duration = s.get("duration", 0.0)
            out.append({"text": text, "start": float(start),
                        "duration": float(duration)})
        return out
    except Exception as e:
        print(f"  ! instance fetch failed for {video_id}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def transcript_to_text(transcript: list[dict]) -> str:
    lines = []
    for entry in transcript:
        ts = format_timestamp(entry["start"])
        text = entry["text"].replace("\n", " ").strip()
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls_file", type=Path,
                        help="Text file with one YouTube URL per line "
                             "(# comments and blank lines ignored).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("docs/ben_transcripts"),
                        help="Directory to write transcripts into.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if the output file already exists.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.urls_file.read_text(encoding="utf-8").splitlines()
    urls = [line.strip() for line in raw
            if line.strip() and not line.strip().startswith("#")]
    print(f"Found {len(urls)} URLs in {args.urls_file}")

    summary = []
    for url in urls:
        vid = extract_video_id(url)
        if not vid:
            print(f"  ! skipping malformed URL: {url}")
            summary.append((url, "?", "BAD_URL"))
            continue

        out_path = args.out_dir / f"{vid}.txt"
        if out_path.exists() and not args.force:
            print(f"= {vid} already cached at {out_path} (use --force to redo)")
            summary.append((vid, "(cached)", "CACHED"))
            continue

        print(f"Fetching {vid} ...")
        meta = fetch_metadata(url)
        transcript = fetch_transcript(vid)
        if transcript is None or len(transcript) == 0:
            print(f"  ! no transcript available for {vid}")
            summary.append((vid, meta.get("title", "?"), "FAILED"))
            continue

        header = (
            f"# {meta.get('title', '(unknown title)')}\n"
            f"# Author: {meta.get('author_name', '')}\n"
            f"# URL: {url}\n"
            f"# Video ID: {vid}\n"
            f"# Snippets: {len(transcript)}\n"
            f"\n"
        )
        out_path.write_text(header + transcript_to_text(transcript),
                            encoding="utf-8")
        print(f"  -> {out_path}  ({len(transcript)} snippets)")
        summary.append((vid, meta.get("title", "?"), "OK"))

    print("\n=== Summary ===")
    for vid, title, status in summary:
        title_short = (title[:70] + "...") if len(title) > 73 else title
        print(f"  [{status:>7}] {vid}  {title_short}")

    failures = [s for s in summary if s[2] in ("FAILED", "BAD_URL")]
    if failures:
        print(f"\n{len(failures)} video(s) failed — see messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
