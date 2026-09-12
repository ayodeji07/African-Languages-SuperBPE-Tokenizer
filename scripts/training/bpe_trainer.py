"""
From-scratch, two-stage BPE/SuperBPE trainer.

Stage 1 (word-respecting): train on whitespace-split "words" -- standard
BPE, merges never cross a space, exactly like any production BPE tokenizer.

Stage 2 (SuperBPE / whitespace-crossing): continue training, but now the
training unit is a whole *line*, built from stage-1 tokens with the literal
space character re-inserted between former words. Because space is just
another symbol at this point, merges are free to combine a word-final token
with an adjacent space and the next word's first token -- this is the
mechanism the SuperBPE paper calls "superword" merging.

Both stages share one growing integer symbol space, so stage 2 continues
directly from wherever stage 1 left off -- no separate vocab reconciliation
needed.

Efficient merge-learning algorithm: a max-heap over pair frequencies with
lazy staleness checking (skip a popped entry if its count no longer matches
the authoritative pair_counts dict), and updates scoped only to the
sequences known to contain the pair being merged -- this avoids rescanning
the whole corpus on every merge, which is what makes 20,000 merges over
millions of sequences tractable in pure Python.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path


def fingerprint_unit_freqs(unit_freqs: dict[tuple[int, ...], int]) -> str:
    """Cheap sanity check, not a cryptographic hash: catches 'you're resuming
    a checkpoint against different data than it was written for' mistakes,
    which would otherwise silently replay merges onto the wrong sequences
    and produce a corrupted-but-not-crashing vocabulary -- the worst failure
    mode, same category as the tab-character bug from the submission pilot.
    """
    n_keys = len(unit_freqs)
    total_freq = sum(unit_freqs.values())
    total_symbols = sum(len(k) * f for k, f in unit_freqs.items())
    return f"{n_keys}:{total_freq}:{total_symbols}"


class SymbolTable:
    def __init__(self):
        self.symbol_str: list[str] = []
        self._base_ids: dict[str, int] = {}

    def seed_base_alphabet(self, chars: str):
        for ch in chars:
            self.get_base_id(ch)

    def get_base_id(self, ch: str) -> int:
        if ch not in self._base_ids:
            self._base_ids[ch] = len(self.symbol_str)
            self.symbol_str.append(ch)
        return self._base_ids[ch]

    def new_merged_id(self, a: int, b: int) -> int:
        new_id = len(self.symbol_str)
        self.symbol_str.append(self.symbol_str[a] + self.symbol_str[b])
        return new_id

    def display(self, sym_id: int) -> str:
        return self.symbol_str[sym_id]


def save_snapshot(path: Path, merges: list[tuple[int, int, int]], fingerprint: str, target_vocab_size: int):
    """Lightweight: merges only. For the training-curve feature, which only
    ever needs to rebuild a vocab/merge_rank to encode FRESH text -- never
    needs the training corpus's internal sequence state.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "merges": merges,
        "fingerprint": fingerprint,
        "target_vocab_size": target_vocab_size,
    }), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem -- never leaves a half-written file


def save_checkpoint(path: Path, merges: list[tuple[int, int, int]], fingerprint: str,
                     target_vocab_size: int, seqs: list[list[int]]):
    """Heavyweight: includes the current per-line/per-word compressed
    sequence state, not just the list of merges. This is what makes resume
    cheap -- without it, resuming has to redo the full initial pair-counting
    pass over the RAW corpus AND reapply every checkpointed merge one by one
    to reconstruct this same state, which costs close to what it cost the
    first time (confirmed in practice: a resume was taking comparably long
    to the original run it was resuming). With seqs saved directly, resuming
    is one cheap pass over already-compressed sequences, not a full redo.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "merges": merges,
        "fingerprint": fingerprint,
        "target_vocab_size": target_vocab_size,
        "seqs": seqs,
    }), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem -- never leaves a half-written checkpoint


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["merges"] = [tuple(m) for m in data["merges"]]
    return data


def train_bpe(
    unit_freqs: dict[tuple[int, ...], int],
    target_vocab_size: int,
    table: SymbolTable,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 500,
    snapshot_dir: Path | None = None,
    snapshot_every: int = 1000,
    max_span_words: int | None = None,
    merge_criterion: str = "frequency",
    pmi_min_count: int = 5,
    pmi_rebuild_every: int = 300,
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, ...], list[int]]]:
    """
    unit_freqs: mapping from an initial symbol-id sequence (a "word" for
        stage 1, a "line" for stage 2) to how many times it occurs.
    target_vocab_size: stop once table has this many total symbols.
    table: shared SymbolTable, mutated in place as merges are learned.
    checkpoint_path: if given, progress is saved here every checkpoint_every
        merges (atomically), and resumed from automatically if the file
        already exists and its fingerprint matches this exact unit_freqs --
        a mismatch raises rather than silently training on the wrong replay,
        since a corrupted-but-not-crashing vocabulary is the worst failure
        mode (same lesson as the tab-character escape-hatch bug).
    snapshot_dir: if given, an additional copy of progress is written every
        snapshot_every merges to a VERSIONED filename (unlike checkpoint_path,
        which only ever keeps the latest). This is for building a
        compression-vs-vocab-size curve after training, not for resuming --
        cheap (reuses the same save format), doesn't slow the hot loop.

    merge_criterion: "frequency" (standard BPE -- always pick the pair with
        the highest raw co-occurrence count) or "pmi" (WordPiece-style --
        pick the pair maximizing log(count(a,b)) - log(count(a)) - log(count(b)),
        favoring pairs that co-occur far more than their individual
        frequencies would predict, not just pairs that are common in
        absolute terms). PMI mode deliberately does NOT try to maintain a
        perfectly live-updated heap: a pair's PMI score depends on the
        GLOBAL occurrence counts of its two symbols, which can shift from
        ANY merge touching either symbol anywhere in the corpus, not just
        merges touching that exact pair -- tracking that precisely would
        need a second reverse index (symbol -> every pair containing it),
        adding real complexity for a scheme already burned once this
        project on a subtle heap-staleness bug. Instead, PMI mode rebuilds
        the heap from scratch (guaranteed correct at that instant) every
        pmi_rebuild_every merges and whenever it empties out, accepting a
        small, bounded staleness window between rebuilds in exchange for
        much simpler, easier-to-trust code.
    pmi_min_count: pairs with fewer than this many occurrences are excluded
        from PMI ranking entirely -- pure PMI over-rewards pairs that are
        rare almost by chance (two low-frequency symbols that happened to
        co-occur every time they appeared), so this floor keeps PMI from
        chasing noise.

    Returns (merges, final_seqs) where merges is an ordered list of
    (a, b, new_id) and final_seqs maps each ORIGINAL input key to its fully
    merged representation -- used to build stage 2's input directly without
    re-running merge application from scratch.
    """
    import math

    keys = list(unit_freqs.keys())
    freqs = [unit_freqs[k] for k in keys]
    fingerprint = fingerprint_unit_freqs(unit_freqs)

    merges: list[tuple[int, int, int]] = []
    start_vocab = len(table.symbol_str)

    # ---- resume: restore the actual sequence state directly (cheap: just
    # deserializing), instead of rebuilding it from raw input + replaying
    # every merge (expensive: costs close to what the original run cost,
    # confirmed in practice -- a resume was taking comparably long to redo
    # as it took to learn the first time). table.symbol_str still needs
    # fast-forwarding, but that's just list appends, not real recomputation.
    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None:
        if ckpt["fingerprint"] != fingerprint:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} was written for different training data "
                f"(fingerprint {ckpt['fingerprint']} != current {fingerprint}). Refusing to "
                f"resume against mismatched data -- delete the checkpoint if this is intentional."
            )
        print(f"    resuming from checkpoint: {len(ckpt['merges']):,} merges already done "
              f"(state restored directly, not replayed)", flush=True)
        for a, b, new_id in ckpt["merges"]:
            got_id = table.new_merged_id(a, b)
            assert got_id == new_id, "checkpoint replay produced a different symbol id than recorded"
        merges = list(ckpt["merges"])
        if "seqs" in ckpt:
            seqs = [list(s) for s in ckpt["seqs"]]
        else:
            # older checkpoint format without saved state -- fall back to the
            # slow path (rebuild raw + reapply) for this one resume only
            print("    (old-format checkpoint without saved state -- falling back "
                  "to slower reconstruction just this once)", flush=True)
            seqs = [list(k) for k in keys]
            tmp_pair_counts: Counter = Counter()
            tmp_pair_to_idxs: dict[tuple[int, int], set[int]] = defaultdict(set)
            for i, seq in enumerate(seqs):
                f = freqs[i]
                for a, b in zip(seq, seq[1:]):
                    tmp_pair_counts[(a, b)] += f
                    tmp_pair_to_idxs[(a, b)].add(i)
            for a, b, new_id in ckpt["merges"]:
                idxs = tmp_pair_to_idxs.pop((a, b), ())
                for i in idxs:
                    seq = seqs[i]
                    new_seq = []
                    j = 0
                    while j < len(seq):
                        if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                            new_seq.append(new_id)
                            j += 2
                        else:
                            new_seq.append(seq[j])
                            j += 1
                    seqs[i] = new_seq
    else:
        seqs = [list(k) for k in keys]

    pair_counts: Counter = Counter()
    pair_to_idxs: dict[tuple[int, int], set[int]] = defaultdict(set)
    symbol_counts: Counter = Counter()

    for i, seq in enumerate(seqs):
        f = freqs[i]
        for a, b in zip(seq, seq[1:]):
            pair_counts[(a, b)] += f
            pair_to_idxs[(a, b)].add(i)
        for s in seq:
            symbol_counts[s] += f

    def pmi_score(pair: tuple[int, int]) -> float:
        c = pair_counts.get(pair, 0)
        if c <= 0:
            return float("-inf")
        a, b = pair
        return math.log(c) - math.log(max(symbol_counts.get(a, 1), 1)) - math.log(max(symbol_counts.get(b, 1), 1))

    def rebuild_heap():
        if merge_criterion == "pmi":
            h = [(-pmi_score(p), p) for p, c in pair_counts.items() if c >= pmi_min_count]
        else:
            h = [(-c, p) for p, c in pair_counts.items() if c > 0]
        heapq.heapify(h)
        return h

    heap = rebuild_heap()
    pushes_since_rebuild = 0
    merges_since_pmi_rebuild = 0
    REBUILD_EVERY = 300_000  # bounds heap growth -- without this it grows unboundedly over 10k+ merges

    def apply_one_merge(a: int, b: int, new_id: int):
        nonlocal pushes_since_rebuild, heap
        idxs = pair_to_idxs.pop((a, b), ())
        # Every pair whose count changes (up OR down) needs a fresh heap entry
        # reflecting its CURRENT value -- a pair whose count only ever goes
        # down after some point (never up again) would otherwise have no
        # valid heap entry pointing at it: old entries at the higher count
        # are correctly recognized as stale and skipped, but nothing reflects
        # the new lower count either, so the pair silently becomes
        # unreachable even though pair_counts still correctly tracks it.
        # (Confirmed empirically: a pair with the objectively highest count
        # was skipped in favor of a lower one because its only recent change
        # was a pure decrease.) But pushing once per OCCURRENCE is wasteful
        # when a high-frequency pair (e.g. anything adjacent to space) is
        # touched across tens of thousands of sequences in one merge call --
        # dedup via `touched` and push once per unique pair, using its final
        # post-merge count, instead of once per occurrence.
        touched: set[tuple[int, int]] = set()
        for i in idxs:
            seq = seqs[i]
            if len(seq) < 2:
                continue
            f = freqs[i]
            for x, y in zip(seq, seq[1:]):
                pair_counts[(x, y)] -= f
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
                    pair_to_idxs.pop((x, y), None)
                touched.add((x, y))

            new_seq = []
            j = 0
            n_merged_here = 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    new_seq.append(new_id)
                    j += 2
                    n_merged_here += 1
                else:
                    new_seq.append(seq[j])
                    j += 1
            seqs[i] = new_seq
            if n_merged_here:
                c = n_merged_here * f
                symbol_counts[a] -= c
                symbol_counts[b] -= c
                symbol_counts[new_id] += c

            for x, y in zip(new_seq, new_seq[1:]):
                pair_counts[(x, y)] += f
                pair_to_idxs[(x, y)].add(i)
                touched.add((x, y))

        # PMI mode deliberately skips incremental pushes here -- a pair's
        # PMI score depends on the GLOBAL symbol_counts of its two symbols,
        # which can shift from a merge touching either symbol ANYWHERE in
        # the corpus, not just merges touching this exact pair. Pushing
        # only for `touched` pairs would leave others silently stale in
        # the same way the old frequency-only heap once was (see the
        # REMOVE-without-ADD bug this project already hit and fixed).
        # Correctness instead comes from the periodic + on-empty rebuilds
        # in the main loop -- see merge_criterion docstring above.
        if merge_criterion == "frequency":
            for p in touched:
                c = pair_counts.get(p)
                if c is not None and c > 0:
                    heapq.heappush(heap, (-c, p))
                    pushes_since_rebuild += 1

            if pushes_since_rebuild > REBUILD_EVERY:
                heap = rebuild_heap()
                pushes_since_rebuild = 0

    while len(table.symbol_str) < target_vocab_size:
        if not heap:
            # PMI mode intentionally doesn't push incrementally between
            # rebuilds (see apply_one_merge), so the heap can legitimately
            # run dry before pmi_rebuild_every is reached -- rebuild now
            # instead of treating an empty heap as "done".
            heap = rebuild_heap()
            merges_since_pmi_rebuild = 0
            if not heap:
                break

        neg_c, pair = heapq.heappop(heap)
        a, b = pair
        if merge_criterion == "pmi":
            c = pair_counts.get(pair, 0)
            if c < pmi_min_count:
                continue  # below the noise floor, discard
            if abs(pmi_score(pair) - (-neg_c)) > 1e-9:
                continue  # stale heap entry (score has drifted), discard
        else:
            if pair_counts.get(pair, 0) != -neg_c or -neg_c <= 0:
                continue  # stale heap entry, discard
        if max_span_words is not None:
            # word_span = (number of literal spaces in the merged string) + 1.
            # Mirrors the SuperBPE paper's cap on how many words a single
            # "superword" token may bridge (they use 4), to stop budget being
            # spent on overly-specific long spans instead of more broadly
            # reusable shorter ones. This pair is permanently unmergeable --
            # just skip this heap entry, same as a stale one.
            span = table.symbol_str[a].count(" ") + table.symbol_str[b].count(" ") + 1
            if span > max_span_words:
                continue

        new_id = table.new_merged_id(a, b)
        merges.append((a, b, new_id))
        apply_one_merge(a, b, new_id)

        if merge_criterion == "pmi":
            merges_since_pmi_rebuild += 1
            if merges_since_pmi_rebuild >= pmi_rebuild_every:
                heap = rebuild_heap()
                merges_since_pmi_rebuild = 0

        if checkpoint_path and len(merges) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, merges, fingerprint, target_vocab_size, seqs)

        if snapshot_dir and len(merges) % snapshot_every == 0:
            snap_path = snapshot_dir / f"vocab_{len(table.symbol_str):06d}.json"
            save_snapshot(snap_path, merges, fingerprint, target_vocab_size)

        if len(merges) % 1000 == 0:
            done = len(table.symbol_str) - start_vocab
            print(f"    ...{done:,} merges done, vocab={len(table.symbol_str):,}, "
                  f"heap={len(heap):,}, pairs={len(pair_counts):,}", flush=True)

    if checkpoint_path:
        save_checkpoint(checkpoint_path, merges, fingerprint, target_vocab_size, seqs)
    if snapshot_dir:
        save_snapshot(snapshot_dir / f"vocab_{len(table.symbol_str):06d}.json",
                       merges, fingerprint, target_vocab_size)

    final_seqs = {keys[i]: seqs[i] for i in range(len(keys))}
    return merges, final_seqs


def apply_merges(symbols: list[int], merge_rank: dict[tuple[int, int], tuple[int, int]]) -> list[int]:
    """Efficient greedy BPE encode: O(n log n) via a doubly-linked list over
    positions + a lazily-validated min-heap of candidate merges, instead of
    rescanning the whole sequence on every merge (which is O(n^2) and, at
    real competition scale -- 2M characters, marker-heavy text expanding
    9-23x -- would blow the 20-minute runtime budget).
    """
    n = len(symbols)
    if n < 2:
        return list(symbols)

    ids = list(symbols)
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    alive = [True] * n
    heap: list[tuple[int, int, int, int, int, int]] = []

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
            continue  # stale: positions changed or already merged since this was pushed
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


def build_merge_rank(merges: list[tuple[int, int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    return {(a, b): (rank, new_id) for rank, (a, b, new_id) in enumerate(merges)}
