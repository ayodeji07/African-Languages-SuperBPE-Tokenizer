"""
Turn the versioned snapshots written by train_bpe (snapshot_dir) into a
compression-vs-vocab-size curve: tokens/char on held-out text AND on a
training-data sample, at each vocab-size checkpoint, for both stages.

This is the BPE-appropriate analog of a train/validation loss curve --
there's no gradient descent to plot, but this shows the same thing a loss
curve is for: whether held-out compression keeps pace with training
compression (no overfitting) or lags behind/plateaus while training keeps
improving (overfitting), and whether either curve is still steeply
improving at the vocab cap (underfitting -- more budget would still help).

Usage:
    python collect_training_curve.py \
        --snapshot-dir data/full_corpus/snapshots \
        --stage1-complete data/full_corpus/tokenizer.stage1_complete.json \
        --heldout-dir data/pilot_cache \
        --train-sample-dir data/full_corpus/train_samples \
        --out training_curve.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe_trainer import apply_merges, build_merge_rank, load_checkpoint  # noqa: E402


def load_symbol_str_and_merges(stage1_complete: dict, stage2_merges: list[tuple[int, int, int]] | None):
    """Reconstruct symbol_str + a usable merge_rank for a given point in
    training. stage2_merges=None means "stage-1-only snapshot". Reconstructing
    symbol_str is cheap (just list appends) -- no need to touch sequences.
    """
    from bpe_trainer import SymbolTable
    table = SymbolTable()
    table.seed_base_alphabet("".join(chr(i) for i in range(32, 127)))
    all_merges = list(stage1_complete["merges"])
    if stage2_merges:
        all_merges += list(stage2_merges)
    for a, b, new_id in all_merges:
        got = table.new_merged_id(a, b)
        assert got == new_id, "symbol id mismatch while reconstructing snapshot vocab"
    return table.symbol_str, build_merge_rank(all_merges)


def measure(symbol_str: list[str], merge_rank: dict, text_by_lang: dict[str, list[str]]) -> dict:
    # Encode line-by-line, matching how validate_submission.py and the real
    # evaluator call encode() -- never as one blob with raw embedded
    # newlines. ord('\n')-32 is -22, a negative index into symbol_str that
    # silently aliases to an unrelated vocab entry instead of erroring (the
    # same class of bug fixed in the submission's tokenizer.py escape hatch).
    total_tokens = total_chars = 0
    per_lang = {}
    for lang, lines in text_by_lang.items():
        lang_tokens = lang_chars = 0
        for line in lines:
            ids = [ord(ch) - 32 for ch in line]
            encoded = apply_merges(ids, merge_rank)
            lang_tokens += len(encoded)
            lang_chars += len(line)
        per_lang[lang] = lang_tokens / max(lang_chars, 1)
        total_tokens += lang_tokens
        total_chars += lang_chars
    return {"overall": total_tokens / max(total_chars, 1), "per_lang": per_lang}


def load_text_dir(d: Path, suffix_to_strip: str) -> dict[str, list[str]]:
    out = {}
    for p in sorted(d.glob("*.txt")):
        lang = p.stem.replace(suffix_to_strip, "")
        out[lang] = [ln for ln in p.read_text(encoding="utf-8").split("\n") if ln]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True,
                         help="directory containing stage1/ and stage2/ snapshot subfolders")
    parser.add_argument("--stage1-complete", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True,
                         help="directory of {lang}_heldout_processed.txt files")
    parser.add_argument("--train-sample-dir", type=Path, required=True,
                         help="directory of {lang}_train_sample.txt files (a slice of the "
                              "actual training corpus, held fixed for comparison)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stage1_complete = json.loads(args.stage1_complete.read_text(encoding="utf-8"))
    stage1_complete["merges"] = [tuple(m) for m in stage1_complete["merges"]]

    heldout = load_text_dir(args.heldout_dir, "_heldout_processed")
    train_sample = load_text_dir(args.train_sample_dir, "_train_sample")
    print(f"loaded {len(heldout)} held-out files, {len(train_sample)} train-sample files")

    points = []

    for stage, stage2_merges in [("stage1", None)]:
        stage_dir = args.snapshot_dir / "stage1"
        for snap_path in sorted(stage_dir.glob("vocab_*.json")):
            ckpt = load_checkpoint(snap_path)
            vocab_size = int(snap_path.stem.split("_")[1])
            symbol_str, merge_rank = load_symbol_str_and_merges(
                {"merges": ckpt["merges"]}, None
            )
            train_stats = measure(symbol_str, merge_rank, train_sample)
            heldout_stats = measure(symbol_str, merge_rank, heldout)
            points.append({
                "stage": "stage1", "vocab_size": vocab_size,
                "train_tokens_per_char": train_stats["overall"],
                "heldout_tokens_per_char": heldout_stats["overall"],
                "train_per_lang": train_stats["per_lang"],
                "heldout_per_lang": heldout_stats["per_lang"],
            })
            print(f"  stage1 vocab={vocab_size:,}: train={train_stats['overall']:.4f} "
                  f"heldout={heldout_stats['overall']:.4f}", flush=True)

    stage2_dir = args.snapshot_dir / "stage2"
    for snap_path in sorted(stage2_dir.glob("vocab_*.json")):
        ckpt = load_checkpoint(snap_path)
        vocab_size = int(snap_path.stem.split("_")[1])
        symbol_str, merge_rank = load_symbol_str_and_merges(stage1_complete, ckpt["merges"])
        train_stats = measure(symbol_str, merge_rank, train_sample)
        heldout_stats = measure(symbol_str, merge_rank, heldout)
        points.append({
            "stage": "stage2", "vocab_size": vocab_size,
            "train_tokens_per_char": train_stats["overall"],
            "heldout_tokens_per_char": heldout_stats["overall"],
            "train_per_lang": train_stats["per_lang"],
            "heldout_per_lang": heldout_stats["per_lang"],
        })
        print(f"  stage2 vocab={vocab_size:,}: train={train_stats['overall']:.4f} "
              f"heldout={heldout_stats['overall']:.4f}", flush=True)

    args.out.write_text(json.dumps(points, indent=2), encoding="utf-8")
    print(f"saved {len(points)} points -> {args.out}")


if __name__ == "__main__":
    main()
