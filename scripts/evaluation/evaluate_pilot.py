"""Measure per-language compression (tokens per processed character) on the
held-out slices for one or more trained pilot tokenizers, so different
alpha values can be compared on real numbers instead of theory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

DATA_DIR = Path("./data_workspace")
LANG_NAMES = [
    "Amharic", "Hausa", "Somali", "Swahili", "Yoruba", "Igbo", "Kinyarwanda",
    "Nigerian Pidgin", "Luo", "Kanuri", "Dinka", "Nama",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_paths", nargs="+", type=Path,
                         help="one or more trained tokenizer.json files, e.g. from different alpha runs")
    args = parser.parse_args()

    tokenizers_by_path = {p: Tokenizer.from_file(str(p)) for p in args.tokenizer_paths}

    heldout = {}
    for lang in LANG_NAMES:
        slug = lang.lower().replace(" ", "_")
        path = DATA_DIR / "pilot_cache" / f"{slug}_heldout_processed.txt"
        heldout[lang] = path.read_text(encoding="utf-8")

    col_width = 16
    header = f"{'Language':<14}" + "".join(f"{p.parent.name:>{col_width}}" for p in args.tokenizer_paths)
    print(header)
    print("-" * len(header))

    worst_per_run = {p: (None, -1.0) for p in args.tokenizer_paths}
    for lang in LANG_NAMES:
        text = heldout[lang]
        row = f"{lang:<14}"
        for p in args.tokenizer_paths:
            tok = tokenizers_by_path[p]
            n_tokens = len(tok.encode(text).ids)
            ratio = n_tokens / max(len(text), 1)
            row += f"{ratio:>{col_width}.4f}"
            if ratio > worst_per_run[p][1]:
                worst_per_run[p] = (lang, ratio)
        print(row)

    print()
    for p in args.tokenizer_paths:
        vocab_size = tokenizers_by_path[p].get_vocab_size()
        worst_lang, worst_ratio = worst_per_run[p]
        print(f"{p.parent.name}: vocab_size={vocab_size}, worst-compressed language = "
              f"{worst_lang} ({worst_ratio:.4f} tokens/char)")


if __name__ == "__main__":
    main()
