"""Fetch raw isiXhosa (promoted to Tier-1) and Zulu (new Tier-2 canary,
replacing isiXhosa) samples from WURA -- same source and method as
fetch_wura_samples.py. Run on the Lightning Studio, not locally.
"""
import urllib.request
from pathlib import Path

RAW_DIR = Path("./data_workspace/raw_samples")
TIER2_RAW_DIR = Path("./data_workspace/tier2_raw_samples")
RAW_DIR.mkdir(parents=True, exist_ok=True)
TIER2_RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/castorini/wura/resolve/main/passages-v1.0/train/{code}.txt"

# isiXhosa: promoted to Tier-1, needs a large pool like the other
# medium/high-resource WURA languages (hausa/somali/swahili are ~42-47MB raw).
JOBS = [
    ("xho", "xhosa", RAW_DIR, 45_000_000),
    ("zul", "zulu", TIER2_RAW_DIR, 8_000_000),
]


def fetch_range(url: str, num_bytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def main():
    for code, name, out_dir, sample_bytes in JOBS:
        url = BASE_URL.format(code=code)
        data = fetch_range(url, sample_bytes)
        idx = data.rfind(b"\n")
        if idx > 0:
            data = data[:idx]
        out_path = out_dir / f"{name}_raw.txt"
        out_path.write_bytes(data)
        num_lines = data.count(b"\n") + 1
        print(f"{name:<10} {len(data):>10,} bytes  {num_lines:>7,} lines  -> {out_path}")


if __name__ == "__main__":
    main()
