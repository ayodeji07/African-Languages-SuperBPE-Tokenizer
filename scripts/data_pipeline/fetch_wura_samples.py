"""Download small raw-text samples (via HTTP range requests) for each WURA
Tier-1 language, without pulling the full multi-GB files. Run this on the
Lightning Studio, not locally.
"""
import urllib.request
from pathlib import Path

OUT_DIR = Path("./data_workspace/raw_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGS = {
    "amh": "amharic",
    "hau": "hausa",
    "som": "somali",
    "swa": "swahili",
    "yor": "yoruba",
    "ibo": "igbo",
    "kin": "kinyarwanda",
}

SAMPLE_BYTES = 5_000_000
BASE_URL = "https://huggingface.co/datasets/castorini/wura/resolve/main/passages-v1.0/train/{code}.txt"


def fetch_range(url: str, num_bytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def main():
    for code, name in LANGS.items():
        url = BASE_URL.format(code=code)
        data = fetch_range(url, SAMPLE_BYTES)
        idx = data.rfind(b"\n")
        if idx > 0:
            data = data[:idx]
        out_path = OUT_DIR / f"{name}_raw.txt"
        out_path.write_bytes(data)
        num_lines = data.count(b"\n") + 1
        print(f"{name:<14} {len(data):>10,} bytes  {num_lines:>6,} lines  -> {out_path}")


if __name__ == "__main__":
    main()
