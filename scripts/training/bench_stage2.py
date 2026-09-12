"""Quick benchmark: time the first N stage-2 merges on the real corpus, to
gauge whether the dedup fix restores acceptable throughput before committing
to a full 12,000-merge run.
"""
import json
import time
from collections import Counter
from pathlib import Path

from bpe_trainer import SymbolTable, train_bpe

BASE_ALPHABET = "".join(chr(i) for i in range(32, 127))

cached = json.loads(Path("data/full_corpus_100m/tokenizer.stage1_complete.json").read_text(encoding="utf-8"))
table = SymbolTable()
table.symbol_str = cached["symbol_str"]
final_word_seqs = {tuple(k): v for k, v in cached["final_word_seqs"]}
space_id = table.get_base_id(" ") if " " not in table.symbol_str else table._base_ids.get(" ")
# space_id was fixed at table construction time originally; recover it the same way train_superbpe_pilot.py does
tmp = SymbolTable()
tmp.seed_base_alphabet(BASE_ALPHABET)
space_id = tmp.get_base_id(" ")

lines = [ln for ln in Path("data/full_corpus_100m/combined_train.txt").read_text(encoding="utf-8").split("\n") if ln]
print(f"corpus: {len(lines):,} lines")

line_freqs: Counter = Counter()
for line in lines:
    seq = []
    words = line.split(" ")
    for i, word in enumerate(words):
        if word:
            key = tuple(tmp.get_base_id(ch) for ch in word)
            seq.extend(final_word_seqs[key])
        if i != len(words) - 1:
            seq.append(space_id)
    if seq:
        line_freqs[tuple(seq)] += 1

print(f"stage 2: {len(line_freqs):,} unique lines")
print(f"starting vocab: {len(table.symbol_str):,}")

N_MERGES = 300
target = len(table.symbol_str) + N_MERGES
t0 = time.time()
merges, _ = train_bpe(dict(line_freqs), target, table)
elapsed = time.time() - t0
print(f"did {len(merges):,} merges in {elapsed:.1f}s -> {elapsed/max(len(merges),1):.3f} s/merge")
print(f"extrapolated for 12,000 merges: {elapsed/max(len(merges),1)*12000/60:.1f} minutes")
