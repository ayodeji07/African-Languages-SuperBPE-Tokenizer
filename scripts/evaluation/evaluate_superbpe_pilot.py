"""
Evaluate the two-stage SuperBPE pilot tokenizer: per-language compression
(tokens/char) on the same held-out sets used throughout, PLUS the specific
question this whole exercise was for -- what fraction of emitted tokens are
genuine whitespace-crossing "superword" merges, per language. If the small
languages (Luo, Kanuri, Dinka, Nama) show ~0% crossing tokens, that confirms
the concern: they get character/word-level coverage but no superword payoff.

Also does a cheap round-trip sanity check (decode(encode(x)) == x) on a
sample of held-out lines per language, since losslessness is a hard
requirement for the actual submission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe_trainer import apply_merges, build_merge_rank  # noqa: E402

DATA_DIR = Path("./data_workspace")
LANG_NAMES = [
    "Amharic", "Hausa", "Somali", "Swahili", "Yoruba", "Igbo", "Kinyarwanda",
    "Nigerian Pidgin", "Luo", "Kanuri", "Dinka", "Nama",
]


def load(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    symbol_str = d["symbol_str"]
    merges = [tuple(m) for m in d["stage1_merges"]] + [tuple(m) for m in d["stage2_merges"]]
    return symbol_str, build_merge_rank(merges)


def encode(line: str, merge_rank) -> list[int]:
    ids = [ord(ch) - 32 for ch in line]  # base alphabet was seeded chr(32)..chr(126) in order -> id = ord-32
    return apply_merges(ids, merge_rank)


def decode(ids: list[int], symbol_str: list[str]) -> str:
    return "".join(symbol_str[i] for i in ids)


def main():
    tok_path = Path(sys.argv[1])
    symbol_str, merge_rank = load(tok_path)

    print(f"{'Language':<14}{'tokens/char':>13}{'crossing tok %':>16}{'roundtrip':>11}")
    print("-" * 54)
    for lang in LANG_NAMES:
        slug = lang.lower().replace(" ", "_")
        heldout_path = DATA_DIR / "pilot_cache" / f"{slug}_heldout_processed.txt"
        lines = [ln for ln in heldout_path.read_text(encoding="utf-8").split("\n") if ln]

        total_tokens = total_chars = crossing = 0
        roundtrip_ok = True
        for i, line in enumerate(lines):
            ids = encode(line, merge_rank)
            total_tokens += len(ids)
            total_chars += len(line)
            crossing += sum(1 for tid in ids if " " in symbol_str[tid])
            if i < 20:  # spot-check first 20 lines per language for losslessness
                if decode(ids, symbol_str) != line:
                    roundtrip_ok = False

        ratio = total_tokens / max(total_chars, 1)
        crossing_pct = 100 * crossing / max(total_tokens, 1)
        rt = "OK" if roundtrip_ok else "MISMATCH"
        print(f"{lang:<14}{ratio:>13.4f}{crossing_pct:>15.1f}%{rt:>11}", flush=True)


if __name__ == "__main__":
    main()
