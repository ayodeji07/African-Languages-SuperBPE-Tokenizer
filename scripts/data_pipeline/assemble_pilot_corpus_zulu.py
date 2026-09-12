"""
Assemble a small-scale, alpha-mixed, *preprocessed* training corpus for a
BPE pilot run -- plus a held-out evaluation slice per language, reserved
once and shared across every alpha value tested so comparisons are fair.

Reuses the exact measured bytes_per_char / expansion_factor data from
compute_mixing_ratios.py as the single source of truth for sizing math.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starting_kit"))
from compute_mixing_ratios_zulu import LANGUAGES  # noqa: E402
import preporocess_text as pp  # noqa: E402

DATA_DIR = Path("./data_workspace")
HELDOUT_FRACTION = 0.15  # reserved once, never used in any training slice

SOURCE_FILES = {
    "Amharic": DATA_DIR / "filtered_samples/amharic_filtered.txt",
    "Hausa": DATA_DIR / "filtered_samples/hausa_filtered.txt",
    "Somali": DATA_DIR / "filtered_samples/somali_filtered.txt",
    "Swahili": DATA_DIR / "filtered_samples/swahili_filtered.txt",
    "Yoruba": DATA_DIR / "filtered_samples/yoruba_filtered.txt",
    "Igbo": DATA_DIR / "filtered_samples/igbo_filtered.txt",
    "Kinyarwanda": DATA_DIR / "filtered_samples/kinyarwanda_filtered.txt",
    "Xhosa": DATA_DIR / "filtered_samples/xhosa_filtered.txt",
    "Zulu": DATA_DIR / "tier2_filtered_samples/zulu_filtered.txt",
    "Nigerian Pidgin": DATA_DIR / "raw_samples/pidgin_raw.txt",
    "Luo": DATA_DIR / "raw_samples/luo_glot500.txt",
    "Kanuri": DATA_DIR / "raw_samples/kanuri_manga_ebible.txt",
    "Dinka": DATA_DIR / "raw_samples/dinka_ebible.txt",
    "Nama": DATA_DIR / "raw_samples/nama_raw.txt",
}


def get_pool_and_heldout(lang_name: str) -> tuple[str, str]:
    """Split a language's raw source into a training pool (85%) and a
    held-out slice (15%), cached to disk so every alpha run and the final
    evaluation use the *identical* held-out text.
    """
    cache_dir = DATA_DIR / "pilot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = lang_name.lower().replace(" ", "_")
    pool_path = cache_dir / f"{slug}_pool_raw.txt"
    heldout_raw_path = cache_dir / f"{slug}_heldout_raw.txt"
    heldout_processed_path = cache_dir / f"{slug}_heldout_processed.txt"

    if not (pool_path.exists() and heldout_processed_path.exists()):
        raw = SOURCE_FILES[lang_name].read_text(encoding="utf-8", errors="replace")
        split_idx = int(len(raw) * (1 - HELDOUT_FRACTION))
        # split on a newline near the target index so we don't cut a line in half
        nl = raw.find("\n", split_idx)
        split_idx = nl if nl != -1 else split_idx
        pool, heldout = raw[:split_idx], raw[split_idx:]
        pool_path.write_text(pool, encoding="utf-8")
        heldout_raw_path.write_text(heldout, encoding="utf-8")
        # Preprocess PER LINE, not as one blob -- preprocess_text collapses
        # all whitespace (including newlines) to single spaces, so running it
        # on a multi-line blob in one call destroys paragraph/passage
        # boundaries entirely. The real eval corpus preprocesses each
        # paragraph separately (a list of independently-processed strings),
        # so per-line processing here matches that structure -- and keeps
        # stage-2 SuperBPE training units short instead of one giant blob.
        heldout_lines = [pp.preprocess_text(ln) for ln in heldout.split("\n") if ln.strip()]
        heldout_processed_path.write_text("\n".join(heldout_lines), encoding="utf-8")

    return pool_path.read_text(encoding="utf-8"), heldout_processed_path.read_text(encoding="utf-8")


def build_training_slice(pool: str, target_raw_chars: int) -> str:
    """Take a deterministic slice of the pool; tile (repeat) it if the pool
    is smaller than the target -- this IS the upsampling step for the
    capped clean-source languages.
    """
    if len(pool) == 0:
        return ""
    if target_raw_chars <= len(pool):
        return pool[:target_raw_chars]
    repeats = target_raw_chars // len(pool) + 1
    return (pool * repeats)[:target_raw_chars]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--total-processed-chars", type=float, default=15_000_000)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or DATA_DIR / "pilot" / f"alpha_{args.alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = {lang.name: lang.natural_processed_chars ** args.alpha for lang in LANGUAGES}
    total_weight = sum(weights.values())

    combined_parts = []
    print(f"{'Language':<14}{'target raw chars':>18}{'pool size':>14}{'upsampled?':>12}")
    for lang in LANGUAGES:
        share = weights[lang.name] / total_weight
        target_processed = args.total_processed_chars * share
        target_raw = int(target_processed / lang.expansion_factor)

        pool, heldout_processed = get_pool_and_heldout(lang.name)
        train_raw = build_training_slice(pool, target_raw)
        # per-line, not per-blob -- see get_pool_and_heldout for why
        train_lines = [pp.preprocess_text(ln) for ln in train_raw.split("\n") if ln.strip()]
        train_processed = "\n".join(train_lines)

        lang_slug = lang.name.lower().replace(" ", "_")
        (out_dir / f"{lang_slug}_train.txt").write_text(train_processed, encoding="utf-8")
        combined_parts.append(train_processed)

        upsampled = "yes" if target_raw > len(pool) else ""
        print(f"{lang.name:<14}{target_raw:>18,}{len(pool):>14,}{upsampled:>12}")

    (out_dir / "combined_train.txt").write_text("\n".join(combined_parts), encoding="utf-8")
    total_chars = sum(len(p) for p in combined_parts)
    print(f"\nWrote combined training corpus: {total_chars:,} processed chars -> {out_dir / 'combined_train.txt'}")


if __name__ == "__main__":
    main()
