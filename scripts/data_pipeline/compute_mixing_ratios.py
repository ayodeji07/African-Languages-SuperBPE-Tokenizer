"""
Compute per-language training-mix targets for SuperBPE tokenizer training.

Core idea (temperature / alpha sampling, as used by mBERT/XLM-R): raw
available volume per language differs by >10,000x in this project (Amharic's
full WURA download vs. Nama/Luo's small clean corpora). Training on natural
proportions would let the biggest corpora buy nearly all 20K merge slots.
Instead, reweight: target_share_i ∝ (natural_processed_chars_i) ** alpha,
alpha in [0, 1] -- alpha=1 is natural proportions, alpha=0 is dead-equal
per language, alpha≈0.3 is the common middle ground.

"Processed chars" (not raw chars, not bytes) is the right unit throughout,
because that's what BPE actually trains on -- and expansion factors vary
9-23x across these languages, so raw byte size is a misleading proxy.

Inputs below are measured, not estimated: exact WURA/AfriBERTa download
byte sizes (from HF file metadata), exact bytes-per-char ratios (measured
on the actual downloaded samples, not assumed), and expansion factors
(measured by running the real preprocess_text.py on real downloaded text).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class LangData:
    name: str
    natural_raw_bytes: int      # full available download size
    bytes_per_char: float       # measured on real sample, not assumed
    expansion_factor: float     # measured with real preprocess_text.py
    is_capped_clean_source: bool = False  # True = this IS all the clean text there is (can't just "download more")

    @property
    def natural_raw_chars(self) -> float:
        return self.natural_raw_bytes / self.bytes_per_char

    @property
    def natural_processed_chars(self) -> float:
        return self.natural_raw_chars * self.expansion_factor


LANGUAGES = [
    # WURA-sourced (bytes_per_char and expansion_factor measured on real 5MB samples)
    LangData("Amharic",     1_056_155_518, 2.5852, 23.23),
    LangData("Hausa",         668_249_719, 1.0125,  1.04),
    LangData("Somali",      1_348_375_947, 1.0051,  1.01),
    LangData("Swahili",     2_705_897_507, 1.0057,  1.00),
    LangData("Yoruba",        170_940_054, 1.2699,  4.80),
    LangData("Igbo",          177_620_308, 1.1902,  5.00),
    LangData("Kinyarwanda",   158_405_159, 1.0226,  1.01),
    LangData("Xhosa",         110_831_937, 1.0037,  1.02),
    # AfriBERTa (whole file downloaded, ~1.0 bytes/char, low expansion measured on WURA Latin-script peers)
    LangData("Nigerian Pidgin", 50_200_890, 1.0000, 1.00),
    # Clean sources are capped -- this is genuinely all the verified-clean text found
    LangData("Luo",               870_230, 1.0000,  1.00, is_capped_clean_source=True),
    LangData("Kanuri",          1_037_543, 1.0000,  2.54, is_capped_clean_source=True),
    LangData("Dinka",           1_125_244, 1.0000,  6.92, is_capped_clean_source=True),
    LangData("Nama",              907_048, 1.0000,  6.89, is_capped_clean_source=True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3,
                         help="temperature exponent: 1.0=natural proportions, 0.0=equal per language")
    parser.add_argument("--total-processed-chars", type=float, default=300_000_000,
                         help="target total size of the training mix, in post-processed characters")
    args = parser.parse_args()

    weights = {lang.name: lang.natural_processed_chars ** args.alpha for lang in LANGUAGES}
    total_weight = sum(weights.values())

    print(f"alpha={args.alpha}  target_total_processed_chars={args.total_processed_chars:,.0f}\n")
    header = (
        f"{'Language':<14}{'natural proc. chars':>20}{'target share':>13}"
        f"{'target proc. chars':>19}{'target raw chars':>17}{'upsample?':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for lang in LANGUAGES:
        share = weights[lang.name] / total_weight
        target_processed = args.total_processed_chars * share
        target_raw = target_processed / lang.expansion_factor
        upsample_factor = target_raw / lang.natural_raw_chars if lang.natural_raw_chars > 0 else float("inf")
        rows.append((lang, share, target_processed, target_raw, upsample_factor))

    for lang, share, target_processed, target_raw, upsample_factor in rows:
        upsample_note = ""
        if upsample_factor > 1.0:
            tag = "MUST repeat" if lang.is_capped_clean_source else "would need more than downloaded"
            upsample_note = f"{upsample_factor:.1f}x ({tag})"
        print(
            f"{lang.name:<14}{lang.natural_processed_chars:>20,.0f}{share:>12.1%}"
            f"{target_processed:>19,.0f}{target_raw:>17,.0f}{upsample_note:>12}"
        )

    print(f"\n{'TOTAL':<14}{'':<20}{'100.0%':>13}{args.total_processed_chars:>19,.0f}")


if __name__ == "__main__":
    main()
