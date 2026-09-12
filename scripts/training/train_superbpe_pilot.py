"""
Train a two-stage SuperBPE tokenizer on the pilot corpus and save it.

Stage 1 (0..t): word-respecting BPE, exactly like train_bpe_pilot.py's
stage-1-only run -- lets us compare directly against that baseline.

Stage 2 (t..vocab_size): whitespace-crossing "superword" merges, built on
top of stage 1's vocabulary, trained on whole lines with the literal space
character reinstated between former words.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bpe_trainer import SymbolTable, fingerprint_unit_freqs, train_bpe

BASE_ALPHABET = "".join(chr(i) for i in range(32, 127))


def save_stage1_complete(path: Path, symbol_str: list[str], merges: list, final_word_seqs: dict, fingerprint: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "fingerprint": fingerprint,
        "symbol_str": symbol_str,
        "merges": merges,
        # tuple keys aren't valid JSON object keys -- store as [key_list, value_list] pairs
        "final_word_seqs": [[list(k), v] for k, v in final_word_seqs.items()],
    }), encoding="utf-8")
    tmp.replace(path)


def load_stage1_complete(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["final_word_seqs"] = {tuple(k): v for k, v in data["final_word_seqs"]}
    data["merges"] = [tuple(m) for m in data["merges"]]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--stage1-vocab-size", type=int, default=12000)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="save progress every N merges per stage; 0 disables checkpointing")
    parser.add_argument("--snapshot-dir", type=Path, default=None,
                         help="if given, write versioned vocab snapshots here for a "
                              "compression-vs-vocab-size curve after training")
    parser.add_argument("--snapshot-every", type=int, default=1000)
    parser.add_argument("--max-span-words", type=int, default=None,
                         help="cap stage-2 superword merges to spanning at most this many "
                              "words (SuperBPE paper uses 4); None = no cap")
    parser.add_argument("--merge-criterion", choices=["frequency", "pmi"], default="frequency",
                         help="stage-2 merge selection rule: 'frequency' (standard) or "
                              "'pmi' (WordPiece-style, favors cohesive-but-rarer pairs)")
    parser.add_argument("--pmi-min-count", type=int, default=5,
                         help="pmi mode only: minimum pair count to be eligible for PMI ranking")
    parser.add_argument("--pmi-rebuild-every", type=int, default=300,
                         help="pmi mode only: full heap rebuild interval, in merges")
    args = parser.parse_args()

    stage1_ckpt = args.out.with_suffix(".stage1_ckpt.json") if args.checkpoint_every else None
    stage2_ckpt = args.out.with_suffix(".stage2_ckpt.json") if args.checkpoint_every else None
    stage1_complete_path = args.out.with_suffix(".stage1_complete.json")
    if args.snapshot_dir:
        (args.snapshot_dir / "stage1").mkdir(parents=True, exist_ok=True)
        (args.snapshot_dir / "stage2").mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in args.corpus.read_text(encoding="utf-8").split("\n") if ln]
    print(f"corpus: {len(lines):,} lines, {sum(len(l) for l in lines):,} chars")

    table = SymbolTable()
    table.seed_base_alphabet(BASE_ALPHABET)
    space_id = table.get_base_id(" ")

    # ---- stage 1: word-respecting ----
    word_freqs: Counter = Counter()
    for line in lines:
        for word in line.split(" "):
            if word:
                word_freqs[tuple(table.get_base_id(ch) for ch in word)] += 1
    stage1_fingerprint = fingerprint_unit_freqs(word_freqs)

    # If stage 1 already finished on a prior (interrupted) run, skip straight
    # to stage 2 instead of replaying every stage-1 merge again just to
    # rebuild state that never changes once stage 1 is done -- at full scale
    # that replay could itself take a long time, and would be repeated on
    # every restart if stage 2 needs multiple resumes.
    cached = load_stage1_complete(stage1_complete_path)
    if cached is not None and cached["fingerprint"] == stage1_fingerprint:
        print("stage 1: found a completed cache from a prior run, loading it directly "
              "(skipping stage-1 training entirely)")
        table.symbol_str = cached["symbol_str"]
        final_word_seqs = cached["final_word_seqs"]
        stage1_merges = cached["merges"]
        print(f"stage 1: loaded, {len(stage1_merges):,} merges, vocab={len(table.symbol_str):,}")
    else:
        if cached is not None:
            print("stage 1: found a completed cache but it doesn't match this corpus -- "
                  "retraining stage 1 from scratch/checkpoint instead")
        print(f"stage 1: {len(word_freqs):,} unique words, training to vocab_size={args.stage1_vocab_size}")
        stage1_merges, final_word_seqs = train_bpe(
            word_freqs, args.stage1_vocab_size, table,
            checkpoint_path=stage1_ckpt, checkpoint_every=args.checkpoint_every or 500,
            snapshot_dir=(args.snapshot_dir / "stage1") if args.snapshot_dir else None,
            snapshot_every=args.snapshot_every,
        )
        print(f"stage 1 done: {len(stage1_merges):,} merges learned, vocab now {len(table.symbol_str):,}")
        save_stage1_complete(stage1_complete_path, table.symbol_str, stage1_merges, final_word_seqs, stage1_fingerprint)

    # ---- stage 2: whitespace-crossing ----
    line_freqs: Counter = Counter()
    for line in lines:
        seq: list[int] = []
        words = line.split(" ")
        for i, word in enumerate(words):
            if word:
                key = tuple(table.get_base_id(ch) for ch in word)
                seq.extend(final_word_seqs[key])
            if i != len(words) - 1:
                seq.append(space_id)
        if seq:
            line_freqs[tuple(seq)] += 1

    print(f"stage 2: {len(line_freqs):,} unique lines, training to vocab_size={args.vocab_size}")
    stage2_merges, _ = train_bpe(
        line_freqs, args.vocab_size, table,
        checkpoint_path=stage2_ckpt, checkpoint_every=args.checkpoint_every or 500,
        snapshot_dir=(args.snapshot_dir / "stage2") if args.snapshot_dir else None,
        snapshot_every=args.snapshot_every,
        max_span_words=args.max_span_words,
        merge_criterion=args.merge_criterion,
        pmi_min_count=args.pmi_min_count,
        pmi_rebuild_every=args.pmi_rebuild_every,
    )
    print(f"stage 2 done: {len(stage2_merges):,} merges learned, vocab now {len(table.symbol_str):,}")

    out = {
        "symbol_str": table.symbol_str,
        "stage1_merges": stage1_merges,
        "stage2_merges": stage2_merges,
        "space_id": space_id,
        "stage1_vocab_size": args.stage1_vocab_size,
        "vocab_size": args.vocab_size,
    }
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
