"""
Point this at real downloaded corpus text (one file per language) to find out,
from ground truth rather than guesswork, which non-ASCII markers actually
appear -- and how often. Run this on Lightning AI once the WURA/mC4/NLLB/eBible
samples are downloaded there; it never needs the raw text to leave that machine.

It answers three questions per language:
  1. Which [U+.... NAME] markers appear at all (the true codepoint inventory --
     use this to correct/extend enumerate_script_coverage.py's hand-curated
     lists, especially for Luo/Kanuri/Dinka/Nama where conventions are
     under-documented).
  2. Which markers are so rare (e.g. <MIN_COUNT occurrences) that frequency-
     based BPE training won't bother merging them without synthetic upsampling.
  3. What fraction of the post-processed character stream each language's
     marker overhead consumes (the expansion-factor risk from the docx note).

Usage:
    python analyze_marker_coverage.py amharic.txt hausa.txt yoruba.txt ...
    python analyze_marker_coverage.py --dir path/to/per_language_txt_files/
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starting_kit"))
import preporocess_text as pp  # noqa: E402

MARKER_RE = re.compile(r"\[U\+[0-9A-Fa-f]{4,6} [^\]]+\]")
MIN_COUNT = 20  # below this, flag as a fragmentation risk unless upsampled


def analyze_file(path: Path, sample_chars: int | None = None) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if sample_chars:
        raw = raw[:sample_chars]
    processed = pp.preprocess_text(raw)

    markers = MARKER_RE.findall(processed)
    marker_counts = Counter(markers)
    marker_char_total = sum(len(m) for m in markers)

    return {
        "raw_chars": len(raw),
        "processed_chars": len(processed),
        "expansion_factor": len(processed) / max(len(raw), 1),
        "unique_markers": len(marker_counts),
        "marker_char_fraction": marker_char_total / max(len(processed), 1),
        "rare_markers": {m: c for m, c in marker_counts.items() if c < MIN_COUNT},
        "top_markers": marker_counts.most_common(15),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--sample-chars", type=int, default=2_000_000,
                         help="cap per-file read size for a quick pass (default 2M chars)")
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        files.extend(sorted(args.dir.glob("*.txt")))
    if not files:
        parser.error("give one or more text files, or --dir path/to/txt_files/")

    for f in files:
        stats = analyze_file(f, args.sample_chars)
        print(f"\n=== {f.name} ===")
        print(f"  raw chars:            {stats['raw_chars']:,}")
        print(f"  processed chars:      {stats['processed_chars']:,}")
        print(f"  expansion factor:     {stats['expansion_factor']:.2f}x")
        print(f"  unique markers:       {stats['unique_markers']}")
        print(f"  marker char fraction: {stats['marker_char_fraction']:.1%} of processed text")
        if stats["rare_markers"]:
            print(f"  RARE markers (< {MIN_COUNT} occurrences -- fragmentation risk unless upsampled):")
            for m, c in sorted(stats["rare_markers"].items(), key=lambda x: x[1]):
                print(f"    [{c:>3}x] {m}")
        if stats["top_markers"]:
            print("  most frequent markers:")
            for m, c in stats["top_markers"]:
                print(f"    [{c:>6}x] {m}")


if __name__ == "__main__":
    main()
