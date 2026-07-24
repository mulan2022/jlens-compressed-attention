#!/usr/bin/env python3
"""Build the scaled J-lens fitting corpus: N English expository paragraphs.

Source: English Wikipedia *featured articles* (well-developed prose, unlike
random stubs). Two-step: pull the featured-article title list once, then fetch
full plaintext extracts a few titles at a time and mine paragraphs of 60-80
words — same register and length as the original 10-paragraph corpus. At most
MAX_PER_ARTICLE paragraphs per article to preserve topical diversity.
Writes one paragraph per line.

No third-party deps; uses urllib so it runs through the lab proxy.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
N_TARGET = 300
MIN_W, MAX_W = 60, 80
MAX_PER_ARTICLE = 3
BATCH_TITLES = 8

BAD_PAT = re.compile(
    r"may refer to|disambiguation|^\d{4} in |List of| soundtrack$"
    r"|\(\d{4}\)$| census |UTC[+-]|^=|citation needed",
    re.IGNORECASE,
)


def _get(params: dict) -> dict:
    params = dict(params, action="query", format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "jlens-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def featured_titles(limit: int = 1500) -> list[str]:
    titles, cont = [], {}
    while len(titles) < limit:
        data = _get(dict(
            list="categorymembers", cmtitle="Category:Featured articles",
            cmnamespace="0", cmlimit="500", **cont))
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(1)
    return titles[:limit]


def paragraphs(extract: str) -> list[str]:
    out = []
    for para in extract.split("\n"):
        para = " ".join(para.split())
        if not para:
            continue
        n = len(para.split())
        if MIN_W <= n <= MAX_W and not BAD_PAT.search(para) and para[0].isupper():
            out.append(para)
    return out


def main() -> None:
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else N_TARGET
    out_path = sys.argv[2] if len(sys.argv) > 2 else "corpus/main300.txt"
    rng = random.Random(20260724)

    print("fetching featured-article titles ...", flush=True)
    titles = featured_titles()
    rng.shuffle(titles)
    print(f"  {len(titles)} titles", flush=True)

    seen: set[str] = set()
    kept: list[str] = []
    i, backoff = 0, 5
    while len(kept) < n_target and i < len(titles):
        batch = titles[i : i + BATCH_TITLES]
        i += BATCH_TITLES
        try:
            data = _get(dict(
                prop="extracts", explaintext="1", exsectionformat="plain",
                titles="|".join(batch)))
            backoff = 5
        except Exception as e:  # rate limit / proxy hiccup — back off and retry
            print(f"  titles {i}: fetch failed ({e}); sleeping {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            i -= BATCH_TITLES  # retry same batch
            continue
        for page in (data.get("query") or {}).get("pages", {}).values():
            for p in paragraphs(page.get("extract", ""))[:MAX_PER_ARTICLE]:
                key = p[:80]
                if key not in seen:
                    seen.add(key)
                    kept.append(p)
        print(f"  {i} titles scanned: {len(kept)}/{n_target}", flush=True)
        time.sleep(2)

    if len(kept) < n_target:
        raise SystemExit(f"only collected {len(kept)} paragraphs")
    rng.shuffle(kept)
    with open(out_path, "w") as f:
        f.write("\n".join(kept[:n_target]) + "\n")
    print(f"wrote {n_target} paragraphs -> {out_path}")


if __name__ == "__main__":
    main()
