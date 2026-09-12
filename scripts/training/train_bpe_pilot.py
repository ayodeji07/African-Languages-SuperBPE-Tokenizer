"""Train a stage-1 (word-internal only, no whitespace-crossing yet) BPE
tokenizer on an assembled pilot corpus. This deliberately does NOT yet
implement SuperBPE's cross-word merges -- the point of this pilot is to
validate the alpha-mixing/vocab-allocation strategy first, cheaply, before
building the full SuperBPE trainer on top of it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--vocab-size", type=int, default=20000)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=[chr(i) for i in range(32, 127)],
        show_progress=False,
    )
    tokenizer.train([str(args.corpus)], trainer)
    tokenizer.save(str(args.out))
    print(f"{args.corpus.name}: trained vocab_size={tokenizer.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
