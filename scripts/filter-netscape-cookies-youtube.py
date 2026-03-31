#!/usr/bin/env python3
"""
Filter a Netscape cookies.txt to rows whose domain is relevant for YouTube / Google login.

Typical flow:
  yt-dlp --cookies-from-browser chrome --cookies cookies.txt URL --skip-download
  python3 scripts/filter-netscape-cookies-youtube.py cookies.txt > cookies-youtube.txt
  ./scripts/set-ytdlp-cookies-secret.sh cookies-youtube.txt

This greatly reduces size and avoids Fly secrets rollout timeouts.
"""
from __future__ import annotations

import sys

# Cookie "domain" field suffixes we keep (YouTube + Google auth CDNs)
_SUFFIXES = (
    "youtube.com",
    "google.com",
    "googlevideo.com",
    "gstatic.com",
    "googleadservices.com",
    "googleusercontent.com",
    "ggpht.com",
    "ytimg.com",
)


def _normalize_domain(field: str) -> str:
    return field.strip().lstrip(".").lower()


def _keep_domain(domain_field: str) -> bool:
    d = _normalize_domain(domain_field)
    if not d:
        return False
    for suf in _SUFFIXES:
        if d == suf or d.endswith("." + suf):
            return True
    return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: filter-netscape-cookies-youtube.py <cookies.txt> [out.txt]", file=sys.stderr)
        print("  If out.txt omitted, writes to stdout.", file=sys.stderr)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    kept = 0
    skipped = 0
    lines_out: list[str] = []

    with open(in_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                lines_out.append(line)
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                skipped += 1
                continue
            if _keep_domain(parts[0]):
                lines_out.append(line)
                kept += 1
            else:
                skipped += 1

    if not any(
        line.startswith("# Netscape") or "HTTP Cookie" in line[:80]
        for line in lines_out[:5]
    ):
        # Ensure a header so yt-dlp accepts the file
        lines_out.insert(0, "# Netscape HTTP Cookie File\n")

    text = "".join(lines_out)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as o:
            o.write(text)
        print(f"Wrote {out_path}: {kept} cookie lines kept, {skipped} lines skipped.", file=sys.stderr)
    else:
        sys.stdout.write(text)
        print(f"# ({kept} lines kept, {skipped} skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
