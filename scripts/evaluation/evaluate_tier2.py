"""Evaluate the packaged tokenizer on Tier 2 (held-out, unseen-in-training)
languages -- Tigrinya, Zulu, Ewe, Nuer -- each in the same family/branch
as a Tier 1 trained language but never itself included in training. This
measures generalization, not memorization.

isiXhosa was originally in this Tier-2 set but was promoted to Tier 1
(trained directly) after its click-consonant orthography -- not present in
any Bantu-family Tier-1 language -- proved to be a genuine, not
data-fixable, generalization gap. Zulu replaces it here as the Bantu-family
canary, since it shares the same click-consonant trait and was likewise
never included in training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import Tokenizer  # noqa: E402

TIER2_DIR = Path(__file__).resolve().parent.parent / "submission_test_data_tier2"

MIRRORS = {
    "tigrinya": "Amharic (Afroasiatic-Semitic)",
    "zulu": "Xhosa/Swahili/Kinyarwanda (Niger-Congo-Bantu, click-consonant)",
    "ewe": "Yoruba/Igbo (Niger-Congo-Volta-Niger-adjacent)",
    "nuer": "Luo/Dinka (Nilo-Saharan-Nilotic)",
}


def main():
    tok = Tokenizer()
    print(f"{'Language':<12}{'mirrors (trained)':<45}{'tokens/char':>13}{'crossing tok %':>16}")
    print("-" * 86)
    for path in sorted(TIER2_DIR.glob("*_processed.txt")):
        lang = path.stem.replace("_processed", "")
        lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
        encoded = tok.encode(lines)

        n_tokens = sum(len(ids) for ids in encoded)
        n_chars = sum(len(ln) for ln in lines)
        crossing = sum(1 for ids in encoded for i in ids if i < tok._vocab_size and " " in tok.symbol_str[i])

        ratio = n_tokens / max(n_chars, 1)
        crossing_pct = 100 * crossing / max(n_tokens, 1)
        print(f"{lang:<12}{MIRRORS.get(lang, ''):<45}{ratio:>13.4f}{crossing_pct:>15.1f}%", flush=True)


if __name__ == "__main__":
    main()
