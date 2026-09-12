"""
Drop passages/lines that contain letters from a different human-language
script than the target language expects -- e.g. Arabic sentences inside the
"Somali" WURA file, or Bengali/Devanagari/Cyrillic fragments inside "Igbo".

Deliberately narrow: only *alphabetic scripts of other languages* trigger a
drop. Symbols, punctuation, emoji, and combining diacritics are left alone --
those aren't wrong-language contamination, and are cheap to cover via the
shared marker vocabulary regardless of which language's file they turn up in
(dropping every line with a stray bullet or euro sign would throw away good
data for no real benefit).

Usage:
    python filter_script_contamination.py input.txt output.txt --script latin
    python filter_script_contamination.py input.txt output.txt --script ethiopic
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

# Unicode character-name prefixes that indicate a *different* human-language
# alphabet than any Tier-1 language uses. A line containing one of these is
# almost certainly a foreign-language sentence caught up in the crawl, not
# genuine in-language content.
FOREIGN_SCRIPT_PREFIXES = [
    "ARABIC", "BENGALI", "DEVANAGARI", "CYRILLIC", "GREEK", "HEBREW",
    "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "TAMIL", "TELUGU",
    "KANNADA", "GUJARATI", "GURMUKHI", "ORIYA", "MALAYALAM", "SINHALA",
    "MYANMAR", "KHMER", "LAO", "TIBETAN", "GEORGIAN", "ARMENIAN",
]

# Always allowed regardless of target script: script-agnostic diacritics
# that legitimately attach to letters in any script.
ALWAYS_ALLOWED_PREFIXES = ["COMBINING", "MODIFIER LETTER"]

TARGET_SCRIPT_PREFIXES = {
    "latin": ["LATIN"],
    # Amharic text routinely mixes in Latin numerals/loanwords/URLs.
    "ethiopic": ["ETHIOPIC", "LATIN"],
}


def line_is_contaminated(
    line: str, allowed_prefixes: list[str], min_foreign_chars: int = 3
) -> tuple[bool, str]:
    """A line is contaminated only once a *script* accumulates enough hits to
    look like a real foreign sentence. A single stray character is far more
    often a look-alike typo (Greek question mark for ';', Cyrillic 'i' for
    Latin 'i') than genuine wrong-language text -- confirmed empirically:
    real contamination lines carried 146-315 foreign characters each, while
    the false positives were 1-2 character look-alikes.
    """
    counts: dict[str, int] = {}
    for ch in line:
        if ch.isascii():
            continue
        name = unicodedata.name(ch, "")
        if not name:
            continue
        if any(name.startswith(p) for p in ALWAYS_ALLOWED_PREFIXES):
            continue
        if any(name.startswith(p) for p in allowed_prefixes):
            continue
        for p in FOREIGN_SCRIPT_PREFIXES:
            if name.startswith(p):
                counts[p] = counts.get(p, 0) + 1
    for script_name, count in counts.items():
        if count >= min_foreign_chars:
            return True, script_name
    return False, ""


def filter_file(in_path: Path, out_path: Path, script: str, show_examples: int = 2, min_foreign_chars: int = 3):
    allowed = TARGET_SCRIPT_PREFIXES[script]
    total_lines = dropped_lines = total_chars = dropped_chars = 0
    drop_reasons: dict[str, int] = {}
    examples = []

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_lines += 1
            total_chars += len(line)
            contaminated, reason = line_is_contaminated(line, allowed, min_foreign_chars)
            if contaminated:
                dropped_lines += 1
                dropped_chars += len(line)
                script_name = reason.split()[0]
                drop_reasons[script_name] = drop_reasons.get(script_name, 0) + 1
                if len(examples) < show_examples:
                    examples.append((script_name, line.strip()[:160]))
                continue
            fout.write(line)

    return {
        "total_lines": total_lines,
        "dropped_lines": dropped_lines,
        "total_chars": total_chars,
        "dropped_chars": dropped_chars,
        "drop_reasons": drop_reasons,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--script", choices=list(TARGET_SCRIPT_PREFIXES), default="latin")
    parser.add_argument("--min-foreign-chars", type=int, default=3)
    args = parser.parse_args()

    stats = filter_file(args.input, args.output, args.script, min_foreign_chars=args.min_foreign_chars)
    pct_lines = 100 * stats["dropped_lines"] / max(stats["total_lines"], 1)
    pct_chars = 100 * stats["dropped_chars"] / max(stats["total_chars"], 1)
    print(f"{args.input.name}: dropped {stats['dropped_lines']:,}/{stats['total_lines']:,} lines "
          f"({pct_lines:.2f}%), {stats['dropped_chars']:,}/{stats['total_chars']:,} chars ({pct_chars:.2f}%)")
    for script_name, count in sorted(stats["drop_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {script_name}: {count} lines")
    for script_name, text in stats["examples"]:
        print(f"    example ({script_name}): {text}")


if __name__ == "__main__":
    main()
