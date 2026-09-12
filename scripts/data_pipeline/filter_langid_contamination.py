"""
For mined-bitext languages (Dinka, Luo, Kanuri) the Unicode-script filter
isn't enough: contamination here is same-script (Latin-vs-Latin, Arabic-vs-
Arabic), e.g. Hindi/Bollywood lyrics and Indonesian/Vietnamese sentences
mixed into "Kanuri" and "Dinka" mined text.

Strategy: run fastText's lid.176 language-ID model on every line. That model
was NOT trained on Dinka/Luo/Kanuri (they're too low-resource to be in its
176 languages), so genuine in-language text should get low-confidence /
scattered predictions. But it WAS trained on the actual contaminant
languages we're seeing (vi, id, ms, hi, en, ar, zh, ta, bn, ...), so a
high-confidence hit on one of those is strong evidence the line is *not*
in our target language, regardless of what script it's written in.

This is a heuristic, not ground truth -- it will still let through some
contamination (any language lid.176 is also weak on) and may occasionally
drop genuine target-language text that coincidentally resembles a trained
language. Report the drop rate and spot-check the kept/dropped examples
before trusting it for a real training run.

Usage:
    python filter_langid_contamination.py input.txt output.txt \
        --model models/lid.176.bin --threshold 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fasttext

# fastText's lid.176 systematically misfires on these for our target
# languages: Albanian (sq) and Azerbaijani (az) both make heavy use of
# e-with-diaeresis / schwa (e"/e") purely as a surface feature, which
# collides with genuine Dinka/Kanuri orthography -- confirmed on real
# samples (e.g. unmistakably genuine Dinka scripture text scored sq=0.99).
# Esperanto (eo) is a well-known general false-positive trap for short,
# morphologically simple text in any under-resourced language. Treating a
# hit on any of these as real contamination would discard large amounts of
# genuine target-language content, so they're excluded from the drop
# trigger -- lines only get dropped for a confident hit on some other label.
DIACRITIC_CONFUSABLE_LANGS = {"sq", "eo", "az", "fi"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/lid.176.bin"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-examples", type=int, default=10)
    args = parser.parse_args()

    model = fasttext.load_model(str(args.model))

    total = dropped = 0
    dropped_by_lang: dict[str, int] = {}
    examples = []

    with args.input.open(encoding="utf-8", errors="replace") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            text = line.strip().replace("\n", " ")
            if not text:
                fout.write(line)
                continue
            labels, probs = model.predict(text, k=1)
            lang = labels[0].replace("__label__", "")
            prob = float(probs[0])
            if prob >= args.threshold and lang not in DIACRITIC_CONFUSABLE_LANGS:
                dropped += 1
                dropped_by_lang[lang] = dropped_by_lang.get(lang, 0) + 1
                if len(examples) < args.show_examples:
                    examples.append((lang, prob, text[:140]))
                continue
            fout.write(line)

    pct = 100 * dropped / max(total, 1)
    print(f"{args.input.name}: dropped {dropped:,}/{total:,} lines ({pct:.1f}%) "
          f"as confidently non-target-language")
    for lang, count in sorted(dropped_by_lang.items(), key=lambda x: -x[1])[:15]:
        print(f"    {lang}: {count}")
    print("  examples of dropped lines:")
    for lang, prob, text in examples:
        print(f"    [{lang} {prob:.2f}] {text}")


if __name__ == "__main__":
    main()
