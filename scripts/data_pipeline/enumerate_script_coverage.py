"""
Generate synthetic 'coverage supplement' text for the non-ASCII codepoints
each Tier-1 language's orthography needs, so SuperBPE training sees every
marker pattern (from starting_kit/preporocess_text.py) enough times to learn
a merge for it -- independent of how much natural prose you can gather.

Why this exists: preprocess_text.py expands every non-ASCII character into
"<transliteration>[U+XXXX UNICODE NAME]". Measured expansion factors:
Amharic (Ge'ez) ~29x, Yoruba diacritics ~12x, Igbo ~6x, Nama clicks ~6x.
A character that never appears in training text will never get a merge,
and will fragment badly at eval time. Some of these codepoint sets (Ethiopic
syllabary, Nama clicks, Hausa hooked letters) are small and fully enumerable
-- so coverage can be *guaranteed* cheaply instead of hoped for via volume.

Confidence levels:
  CERTAIN   - derived directly from the Unicode standard block/character
              database, not from memory of any specific corpus.
  LIKELY    - well-documented standard orthography, but verify against the
              actual downloaded corpus once available (conventions vary by
              source/era/missionary vs. modern orthography).
  VERIFY    - flagged in the project docx as needing real-corpus confirmation
              (Luo, Kanuri, Dinka orthographic diacritic conventions are not
              as standardized/documented as the others).

Usage:
    python enumerate_script_coverage.py            # print summary table
    python enumerate_script_coverage.py --write out_dir/  # write supplement
                                                            # .txt + manifest.json
                                                            # per language
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starting_kit"))
import preporocess_text as pp  # noqa: E402

pp._get_unidecode = lambda: None  # no network installs; see README note below


def ethiopic_block_chars() -> list[str]:
    """CERTAIN: every assigned codepoint in the core Ethiopic Unicode block.

    This is a safe superset for Amharic -- a handful of these syllables belong
    to Tigrinya/other Ethiopian-script languages rather than Amharic proper,
    but including a syllable Amharic text never actually uses just wastes a
    few unused vocab slots, not a correctness problem. Narrow this down once
    real Amharic corpus samples are available (see analyze_marker_coverage.py).
    """
    chars = []
    for cp in range(0x1200, 0x1380):  # Ethiopic block
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if name:
            chars.append(ch)
    return chars


# CERTAIN: specific codepoints from the Unicode Latin Extended-B / Extended
# Additional / Spacing Modifier Letters blocks, looked up by name.
HAUSA_LETTERS = ["ɓ", "Ɓ", "ɗ", "Ɗ", "ƙ", "Ƙ", "ƴ", "Ƴ"]
#                 b-hook    B-hook    d-hook    D-hook    k-hook   K-hook   y-hook   Y-hook

NAMA_CLICK_LETTERS = ["ǀ", "ǁ", "ǂ", "ǃ"]
#                      dental    lateral   alveolar  (post)alveolar/retroflex
# LIKELY additions actively used in modern Namibian Khoekhoegowab orthography
# for tone/nasalization -- verify against real corpus before finalizing:
NAMA_LIKELY_EXTRAS = ["́", "̀", "̃"]  # combining acute/grave/tilde (tone, nasal vowels)

YORUBA_DOTTED_VOWELS = ["ẹ", "Ẹ", "ọ", "Ọ", "ṣ", "Ṣ"]
#                        e-dot    E-dot     o-dot     O-dot     s-dot     S-dot
YORUBA_TONE_MARKS = ["̀", "́"]  # combining grave (low tone), acute (high tone)
YORUBA_PRECOMPOSED_TONED_VOWELS = list("àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ")
# ^ uppercase forms confirmed present (not just theoretical) by analyze_marker_coverage.py
# run against real WURA Yoruba text -- Ú/Ò/Á/É all appeared with real, non-trivial counts.
YORUBA_TONED_N = ["ń", "ǹ"]  # n-acute, n-grave (syllabic nasal)

IGBO_DOTTED_VOWELS = ["ị", "Ị", "ụ", "Ụ"]  # i-dot, I-dot, u-dot, U-dot
IGBO_N_DOT_ABOVE = ["ṅ", "Ṅ"]  # n with dot above (velar nasal), capital

# VERIFY: documented as used in at least some Kanuri orthographies, but the
# docx flags Kanuri/Luo/Dinka as lacking a curated corpus -- confirm actual
# diacritic usage once real text is downloaded (via analyze_marker_coverage.py)
# rather than trusting this list blindly.
KANURI_LIKELY = ["ŋ", "Ŋ"]  # eng (velar nasal) lower/upper

LANGUAGES = {
    "Amharic": {
        "family": "Afroasiatic - Semitic",
        "confidence": "CERTAIN (full Ethiopic Unicode block)",
        "codepoints": ethiopic_block_chars(),
    },
    "Hausa": {
        "family": "Afroasiatic - Chadic",
        "confidence": "CERTAIN (standard Boko orthography hooked letters)",
        "codepoints": HAUSA_LETTERS,
    },
    "Somali": {
        "family": "Afroasiatic - Cushitic",
        "confidence": "CERTAIN (standard orthography is plain ASCII Latin)",
        "codepoints": [],
    },
    "Swahili": {
        "family": "Niger-Congo - Bantu",
        "confidence": "CERTAIN (standard orthography is plain ASCII Latin)",
        "codepoints": [],
    },
    "Yoruba": {
        "family": "Niger-Congo - Volta-Niger",
        "confidence": "LIKELY (standard orthography; verify tone-mark stacking against real corpus)",
        "codepoints": (
            YORUBA_DOTTED_VOWELS
            + YORUBA_TONE_MARKS
            + YORUBA_PRECOMPOSED_TONED_VOWELS
            + YORUBA_TONED_N
        ),
    },
    "Igbo": {
        "family": "Niger-Congo - Volta-Niger",
        "confidence": "LIKELY (standard orthography; verify tone-mark usage against real corpus)",
        "codepoints": IGBO_DOTTED_VOWELS + IGBO_N_DOT_ABOVE + YORUBA_TONE_MARKS,
    },
    "Kinyarwanda": {
        "family": "Niger-Congo - Bantu",
        "confidence": "LIKELY (standard orthography is plain ASCII Latin)",
        "codepoints": [],
    },
    "Nigerian Pidgin": {
        "family": "English Creole",
        "confidence": "CERTAIN (English-lexified, plain ASCII)",
        "codepoints": [],
    },
    "Luo (Dholuo)": {
        "family": "Nilo-Saharan - Nilotic",
        "confidence": "VERIFY (no confirmed diacritic set -- run analyzer on real corpus)",
        "codepoints": [],
    },
    "Kanuri": {
        "family": "Nilo-Saharan - Saharan",
        "confidence": "VERIFY (eng letter likely; confirm against real corpus)",
        "codepoints": KANURI_LIKELY,
    },
    "Dinka": {
        "family": "Nilo-Saharan - Nilotic",
        "confidence": "VERIFY (known complex diacritic system -- length/breathy voice/tone; run analyzer on real corpus before trusting any hand list)",
        "codepoints": [],
    },
    "Nama (Khoekhoegowab)": {
        "family": "Khoisan - Khoe-Kwadi",
        "confidence": "LIKELY (4 core click letters certain; tone/nasalization marks need corpus verification)",
        "codepoints": NAMA_CLICK_LETTERS + NAMA_LIKELY_EXTRAS,
    },
}


def marker_for(ch: str) -> str:
    return pp.preprocess_text(ch)


def build_supplement_text(codepoints: list[str], repeats: int = 40) -> str:
    """Repeat each codepoint embedded in short word-like ASCII contexts so BPE
    learns the marker merge in typical adjacent-token contexts, not in isolation.
    """
    words = []
    for ch in codepoints:
        token = f"a{ch}b {ch}a {ch}{ch}"
        words.extend([token] * repeats)
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, default=None, help="output directory")
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()

    print(f"{'Language':<24}{'Family':<28}{'#codepoints':>12}  Confidence")
    print("-" * 110)
    total_unique = set()
    for lang, info in LANGUAGES.items():
        cps = info["codepoints"]
        total_unique.update(cps)
        print(f"{lang:<24}{info['family']:<28}{len(cps):>12}  {info['confidence']}")
    print("-" * 110)
    print(f"Union of unique non-ASCII codepoints across all languages: {len(total_unique)}")

    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for lang, info in LANGUAGES.items():
            cps = info["codepoints"]
            slug = lang.lower().split(" ")[0].split("(")[0]
            text = build_supplement_text(cps, args.repeats)
            (args.write / f"{slug}_coverage_supplement.txt").write_text(text, encoding="utf-8")
            manifest[lang] = {
                "family": info["family"],
                "confidence": info["confidence"],
                "num_codepoints": len(cps),
                "codepoints": [
                    {"char_escaped": ch.encode("unicode_escape").decode(), "marker": marker_for(ch)}
                    for ch in cps
                ],
            }
        (args.write / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote per-language supplement .txt files + manifest.json to {args.write}")


if __name__ == "__main__":
    main()
