"""Compare per-language held-out compression across the pilot-scale Zulu
alpha-sweep configs, against the no-Zulu baseline. Flags any language whose
tokens/char gets meaningfully worse when Zulu is added -- this is checking
for a repeat of the Yoruba regression seen when isiXhosa was added.

Run on the Lightning Studio.
"""
from __future__ import annotations

import json
from pathlib import Path

from bpe_trainer import SymbolTable, apply_merges, build_merge_rank

DATA_DIR = Path("./data_workspace")
SWEEP_DIR = DATA_DIR / "sweep_zulu"
HELDOUT_DIR = DATA_DIR / "pilot_cache"

CONFIGS = ["no_zulu_a010", "with_zulu_a0.10_40m", "with_zulu_a0.06_40m", "with_zulu_a0.15_40m"]

LANG_SLUGS = [
    "amharic", "hausa", "somali", "swahili", "yoruba", "igbo",
    "kinyarwanda", "xhosa", "nigerian_pidgin", "luo", "kanuri",
    "dinka", "nama", "zulu",
]


def load_tokenizer(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    table = SymbolTable()
    table.seed_base_alphabet("".join(chr(i) for i in range(32, 127)))
    all_merges = [tuple(m) for m in data["stage1_merges"]] + [tuple(m) for m in data["stage2_merges"]]
    for a, b, new_id in all_merges:
        got = table.new_merged_id(a, b)
        assert got == new_id
    return build_merge_rank(all_merges)


def measure(merge_rank: dict, text: str) -> float:
    if not text:
        return 0.0
    ids = [ord(ch) - 32 for ch in text]
    encoded = apply_merges(ids, merge_rank)
    return len(encoded) / len(text)


def main():
    heldout = {}
    for slug in LANG_SLUGS:
        p = HELDOUT_DIR / f"{slug}_heldout_processed.txt"
        if p.exists():
            heldout[slug] = p.read_text(encoding="utf-8")

    results = {}
    for cfg in CONFIGS:
        merge_rank = load_tokenizer(SWEEP_DIR / cfg / "tokenizer.json")
        results[cfg] = {slug: measure(merge_rank, text) for slug, text in heldout.items()}
        print(f"loaded + measured: {cfg}", flush=True)

    baseline = results["no_zulu_a010"]
    print(f"\n{'Language':<18}{'no_zulu(a.10)':>15}", end="")
    for cfg in CONFIGS[1:]:
        print(f"{cfg:>20}", end="")
    print()

    for slug in LANG_SLUGS:
        if slug not in baseline:
            continue
        base_val = baseline[slug]
        print(f"{slug:<18}{base_val:>15.4f}", end="")
        for cfg in CONFIGS[1:]:
            val = results[cfg].get(slug)
            if val is None:
                print(f"{'--':>20}", end="")
                continue
            pct = 100 * (val - base_val) / base_val if base_val > 0 else 0.0
            flag = " !!" if pct > 5 else ""
            print(f"{val:>12.4f} ({pct:+.1f}%){flag}", end="")
        print()

    out = {"results": results}
    (SWEEP_DIR / "sweep_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {SWEEP_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
