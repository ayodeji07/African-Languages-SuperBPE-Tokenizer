"""Find exactly which merges Yoruba loses when Zulu is added, by diffing
the actual stage-2 merge lists between a Yoruba-healthy config (13 langs,
no Zulu) and a Yoruba-hurt config (14 langs, +Zulu), then showing concrete
before/after tokenizations of the same Yoruba text.

Run on the Lightning Studio.
"""
from __future__ import annotations

import json
from pathlib import Path

from bpe_trainer import SymbolTable, apply_merges, build_merge_rank

DATA_DIR = Path("./data_workspace")

HEALTHY = DATA_DIR / "sweep_zulu/no_zulu_a010/tokenizer.json"
HURT = DATA_DIR / "sweep_zulu/with_zulu_a0.10_40m/tokenizer.json"
YORUBA_HELDOUT = DATA_DIR / "pilot_cache/yoruba_heldout_processed.txt"


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    table = SymbolTable()
    table.seed_base_alphabet("".join(chr(i) for i in range(32, 127)))
    stage1 = [tuple(m) for m in data["stage1_merges"]]
    stage2 = [tuple(m) for m in data["stage2_merges"]]
    all_merges = stage1 + stage2
    for a, b, new_id in all_merges:
        got = table.new_merged_id(a, b)
        assert got == new_id
    return table, stage1, stage2


def main():
    healthy_table, healthy_s1, healthy_s2 = load(HEALTHY)
    hurt_table, hurt_s1, hurt_s2 = load(HURT)

    # Every merged STRING learned in stage 2 (whitespace-crossing) for each config
    healthy_strs = {healthy_table.symbol_str[new_id] for a, b, new_id in healthy_s2}
    hurt_strs = {hurt_table.symbol_str[new_id] for a, b, new_id in hurt_s2}

    # Yoruba text uses ASCII markers like [U+1EB9 ...] for its diacritics --
    # find stage-2 merges that touch a marker AND were learned by the
    # healthy config but NOT the hurt one (i.e. Yoruba-relevant merges that
    # got crowded out).
    lost = healthy_strs - hurt_strs
    lost_marker_merges = sorted((s for s in lost if "[U+" in s), key=len, reverse=True)
    print(f"stage-2 merges learned by healthy but NOT by hurt config: {len(lost):,} total, "
          f"{len(lost_marker_merges):,} contain a marker")
    print("\nlongest lost marker-containing merges (most likely Yoruba-relevant):")
    for s in lost_marker_merges[:25]:
        print(f"  {s!r}")

    gained = hurt_strs - healthy_strs
    gained_marker_merges = sorted((s for s in gained if "[U+" in s), key=len, reverse=True)
    print(f"\nstage-2 merges learned by hurt but NOT by healthy config: {len(gained):,} total, "
          f"{len(gained_marker_merges):,} contain a marker")
    print("\nlongest gained marker-containing merges (likely Zulu/Xhosa-relevant, competing for budget):")
    for s in gained_marker_merges[:25]:
        print(f"  {s!r}")

    # concrete before/after: encode the same Yoruba passage with both
    text = YORUBA_HELDOUT.read_text(encoding="utf-8")
    lines = [ln for ln in text.split("\n") if ln][:1]  # first held-out line
    sample = lines[0] if lines else ""

    healthy_rank = build_merge_rank(healthy_s1 + healthy_s2)
    hurt_rank = build_merge_rank(hurt_s1 + hurt_s2)

    ids = [ord(ch) - 32 for ch in sample]
    healthy_encoded = apply_merges(ids, healthy_rank)
    hurt_encoded = apply_merges(ids, hurt_rank)

    print(f"\nsample Yoruba passage ({len(sample)} chars): {sample[:200]!r}...")
    print(f"healthy encode: {len(healthy_encoded)} tokens")
    print(f"hurt encode:    {len(hurt_encoded)} tokens")
    print("\nhealthy tokens:", [healthy_table.symbol_str[i] for i in healthy_encoded[:40]])
    print("\nhurt tokens:   ", [hurt_table.symbol_str[i] for i in hurt_encoded[:40]])


if __name__ == "__main__":
    main()
