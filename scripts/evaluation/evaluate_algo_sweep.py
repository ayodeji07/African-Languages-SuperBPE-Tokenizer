"""Compare per-language held-out compression across the algorithm-variant
sweep (max-span-words cap, lower transition points) against the existing
t=8000/no-cap baseline. All configs share the same 13-language, alpha=0.10
pilot-scale corpus, so this isolates the effect of the algorithm change
itself from any mixing-ratio effect.

Run on the Lightning Studio.
"""
from __future__ import annotations

import json
from pathlib import Path

from bpe_trainer import SymbolTable, apply_merges, build_merge_rank

DATA_DIR = Path("./data_workspace")
HELDOUT_DIR = DATA_DIR / "pilot_cache"

CONFIGS = {
    "a0.10_baseline": DATA_DIR / "sweep_zulu/no_zulu_a010/tokenizer.json",
    "with_zulu_freq_hurt": DATA_DIR / "sweep_zulu/with_zulu_a0.10_40m/tokenizer.json",
    "with_zulu_pmi": DATA_DIR / "sweep_algo/pmi_test_40m.json",
}

LANG_SLUGS = [
    "amharic", "hausa", "somali", "swahili", "yoruba", "igbo",
    "kinyarwanda", "xhosa", "nigerian_pidgin", "luo", "kanuri",
    "dinka", "nama",
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


def measure(merge_rank: dict, text: str) -> tuple[float, int, int]:
    if not text:
        return 0.0, 0, 0
    ids = [ord(ch) - 32 for ch in text]
    encoded = apply_merges(ids, merge_rank)
    return len(encoded) / len(text), len(encoded), len(text)


def main():
    heldout = {}
    for slug in LANG_SLUGS:
        p = HELDOUT_DIR / f"{slug}_heldout_processed.txt"
        if p.exists():
            heldout[slug] = p.read_text(encoding="utf-8")

    results = {}
    blended = {}
    for name, path in CONFIGS.items():
        merge_rank = load_tokenizer(path)
        per_lang = {}
        total_tokens = total_chars = 0
        for slug, text in heldout.items():
            ratio, n_tokens, n_chars = measure(merge_rank, text)
            per_lang[slug] = ratio
            total_tokens += n_tokens
            total_chars += n_chars
        results[name] = per_lang
        blended[name] = total_tokens / total_chars
        print(f"loaded + measured: {name}  (blended={blended[name]:.4f})", flush=True)

    baseline = results["a0.10_baseline"]
    names = list(CONFIGS.keys())
    print(f"\n{'Language':<18}", end="")
    for name in names:
        print(f"{name:>16}", end="")
    print()
    for slug in LANG_SLUGS:
        if slug not in baseline:
            continue
        base_val = baseline[slug]
        print(f"{slug:<18}", end="")
        for name in names:
            val = results[name][slug]
            pct = 100 * (val - base_val) / base_val if base_val > 0 and name != "a0.10_baseline" else 0.0
            if name == "a0.10_baseline":
                print(f"{val:>16.4f}", end="")
            else:
                print(f"{val:.4f}({pct:+.1f}%)".rjust(16), end="")
        print()

    print(f"\n{'BLENDED':<18}", end="")
    for name in names:
        print(f"{blended[name]:>16.4f}", end="")
    print()

    out = {"results": results, "blended": blended}
    (DATA_DIR / "sweep_algo" / "sweep_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {DATA_DIR / 'sweep_algo' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
