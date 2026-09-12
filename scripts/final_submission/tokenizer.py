"""SuperBPE tokenizer submission for the Codabench competition.

Self-contained: standard library only (json, heapq, pathlib), no external
dependencies -- matches the codalab/codalab-legacy:py312 evaluation image,
which does not include tokenizers/transformers/etc.

The learned vocabulary (base ASCII alphabet + stage-1 word-internal merges
+ stage-2 whitespace-crossing "superword" merges) lives in tokenizer.json,
packaged alongside this file at the ZIP root.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

_MODEL_PATH = Path(__file__).resolve().parent / "tokenizer.json"


class Tokenizer:
    def __init__(self):
        data = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
        self.symbol_str: list[str] = data["symbol_str"]
        merges = [tuple(m) for m in data["stage1_merges"]] + [tuple(m) for m in data["stage2_merges"]]
        # rank = position in this combined list; stage-1 merges all have
        # lower rank than stage-2 merges by construction (stage 1 trained to
        # completion before stage 2 begins), so greedy lowest-rank-first
        # application naturally resolves all word-internal merges before
        # ever applying a whitespace-crossing one -- see bpe_trainer.py for
        # the full argument for why this is exactly equivalent to training.
        self.merge_rank: dict[tuple[int, int], tuple[int, int]] = {
            (a, b): (rank, new_id) for rank, (a, b, new_id) in enumerate(merges)
        }
        # Base alphabet was seeded chr(32)..chr(126) in that exact order
        # during training (see SymbolTable.seed_base_alphabet in
        # bpe_trainer.py), so id = ord(ch) - 32 for every base character.
        self._base_offset = 32
        self._vocab_size = len(self.symbol_str)
        # Escape hatch for any character outside the trained base alphabet
        # (32-126). Preprocessing guarantees this never happens on real
        # competition input -- collapsed whitespace becomes a plain space,
        # non-ASCII becomes a readable-ASCII marker -- but WITHOUT this
        # guard, an out-of-range character silently wraps to a NEGATIVE
        # Python list index (e.g. a raw tab, ord 9, gives 9-32=-23) and
        # aliases to an unrelated real vocab entry instead of erroring,
        # corrupting output instead of failing loudly. This keeps encode/
        # decode exactly lossless for any input, in or out of spec, at
        # zero cost for the in-spec case that's 100% of real usage.
        self._escape_base = self._vocab_size + 1_000_000

    def _char_to_id(self, ch: str) -> int:
        code = ord(ch)
        if 32 <= code <= 126:
            return code - self._base_offset
        return self._escape_base + code

    def _encode_one(self, text: str) -> list[int]:
        n = len(text)
        if n == 0:
            return []
        ids = [self._char_to_id(ch) for ch in text]
        if n < 2:
            return ids

        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(n - 1))
        alive = [True] * n
        heap: list[tuple[int, int, int, int, int, int]] = []
        merge_rank = self.merge_rank

        def push_pair(i: int):
            j = nxt[i]
            if j == -1:
                return
            hit = merge_rank.get((ids[i], ids[j]))
            if hit is not None:
                rank, new_id = hit
                heapq.heappush(heap, (rank, i, j, ids[i], ids[j], new_id))

        for i in range(n - 1):
            push_pair(i)

        while heap:
            rank, i, j, a, b, new_id = heapq.heappop(heap)
            if not alive[i] or not alive[j] or ids[i] != a or ids[j] != b:
                continue
            ids[i] = new_id
            alive[j] = False
            nj = nxt[j]
            nxt[i] = nj
            if nj != -1:
                prv[nj] = i
            p = prv[i]
            if p != -1:
                push_pair(p)
            push_pair(i)

        result = []
        i = 0
        while i != -1:
            if alive[i]:
                result.append(ids[i])
            i = nxt[i]
        return result

    def encode(self, texts: list[str]) -> list[list[int]]:
        return [self._encode_one(t) for t in texts]

    def _id_to_str(self, i: int) -> str:
        if i < self._vocab_size:
            return self.symbol_str[i]
        return chr(i - self._escape_base)

    def decode(self, encoded_texts: list[list[int]]) -> list[str]:
        return ["".join(self._id_to_str(i) for i in ids) for ids in encoded_texts]
