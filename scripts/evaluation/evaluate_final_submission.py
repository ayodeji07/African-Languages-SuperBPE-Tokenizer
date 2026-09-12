"""
Evaluate compression (tokens/char) of the ACTUAL packaged tokenizer.py
against held-out data, per language -- not the separate bpe_trainer.py
implementation used during the sweep. This is what closes the loop: does
the shipped submission achieve the compression numbers we reported, after
the escape-hatch bug fix?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import Tokenizer  # noqa: E402

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "submission_test_data"


def main():
    tok = Tokenizer()
    print(f"vocab size: {len(tok.symbol_str):,}\n")
    print(f"{'Language':<18}{'tokens/char':>13}{'crossing tok %':>16}")
    print("-" * 47)

    grand_tokens = grand_chars = 0
    for path in sorted(TEST_DATA_DIR.glob("*_heldout_processed.txt")):
        lang = path.stem.replace("_heldout_processed", "")
        lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
        encoded = tok.encode(lines)

        n_tokens = sum(len(ids) for ids in encoded)
        n_chars = sum(len(ln) for ln in lines)
        crossing = sum(1 for ids in encoded for i in ids if i < tok._vocab_size and " " in tok.symbol_str[i])

        ratio = n_tokens / max(n_chars, 1)
        crossing_pct = 100 * crossing / max(n_tokens, 1)
        print(f"{lang:<18}{ratio:>13.4f}{crossing_pct:>15.1f}%", flush=True)

        grand_tokens += n_tokens
        grand_chars += n_chars

    print("-" * 47)
    print(f"{'OVERALL (unweighted)':<18}{grand_tokens/max(grand_chars,1):>13.4f}")


if __name__ == "__main__":
    main()
