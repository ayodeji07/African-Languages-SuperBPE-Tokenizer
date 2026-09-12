"""
Validate the submission ZIP contents exactly as the Codabench evaluator
would use them:
    encode_tokenizer = Tokenizer(); encoded = encode_tokenizer.encode(corpus)
    decode_tokenizer = Tokenizer(); decoded = decode_tokenizer.decode(encoded)

Checks:
  1. Vocab size <= 20,000 (hard requirement).
  2. Losslessness: decode(encode(x)) == x, using TWO SEPARATE Tokenizer()
     instances (matching the evaluator's actual usage) across every
     held-out line for all 12 languages -- not a spot check.
  3. Output shape/types match the required interface exactly.
  4. A handful of adversarial edge cases (empty string, single char,
     whitespace-only, all-punctuation, repeated text).
  5. Timing, extrapolated to both the debug corpus (50,000 chars) and the
     final evaluation corpus (2,000,000 chars, 20-minute budget).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import Tokenizer  # noqa: E402

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "submission_test_data"


def check_vocab_size():
    tok = Tokenizer()
    n = len(tok.symbol_str)
    status = "OK" if n <= 20_000 else "FAIL"
    print(f"[{status}] vocab size = {n:,} (limit 20,000)")
    return n <= 20_000


def check_losslessness():
    print("\n--- losslessness (two separate Tokenizer() instances, matching evaluator) ---")
    all_ok = True
    total_lines = 0
    total_chars = 0
    for path in sorted(TEST_DATA_DIR.glob("*_heldout_processed.txt")):
        lang = path.stem.replace("_heldout_processed", "")
        lines = [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]

        encode_tok = Tokenizer()
        decode_tok = Tokenizer()
        encoded = encode_tok.encode(lines)
        decoded = decode_tok.decode(encoded)

        mismatches = [i for i, (orig, rt) in enumerate(zip(lines, decoded)) if orig != rt]
        ok = len(mismatches) == 0
        all_ok = all_ok and ok
        total_lines += len(lines)
        total_chars += sum(len(ln) for ln in lines)
        status = "OK" if ok else f"FAIL ({len(mismatches)} mismatches, e.g. line {mismatches[0]})"
        print(f"  [{status}] {lang}: {len(lines):,} lines, {sum(len(l) for l in lines):,} chars")

    print(f"  total: {total_lines:,} lines, {total_chars:,} chars checked")
    return all_ok


def check_edge_cases():
    print("\n--- edge cases ---")
    cases = [
        "",
        "a",
        " ",
        "   ",
        "!@#$%^&*()_+-=[]{}|;:',.<>/?`~",
        "a" * 500,
        "word " * 200,
        '"quoted text with \'nested\' quotes"',
        "line\twith\ttabs\tif\tany\tsurvive\tpreprocessing",
    ]
    encode_tok = Tokenizer()
    decode_tok = Tokenizer()
    encoded = encode_tok.encode(cases)
    decoded = decode_tok.decode(encoded)
    all_ok = True
    for i, (orig, rt, ids) in enumerate(zip(cases, decoded, encoded)):
        ok = orig == rt
        all_ok = all_ok and ok
        assert isinstance(ids, list) and all(isinstance(x, int) for x in ids), "encode must return list[list[int]]"
        assert isinstance(rt, str), "decode must return list[str]"
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {orig[:40]!r} -> {len(ids)} tokens -> {rt[:40]!r}")
    return all_ok


def check_timing():
    print("\n--- timing ---")
    all_text = []
    for path in sorted(TEST_DATA_DIR.glob("*_heldout_processed.txt")):
        all_text.extend(ln for ln in path.read_text(encoding="utf-8").split("\n") if ln)
    total_chars = sum(len(t) for t in all_text)

    t0 = time.perf_counter()
    encode_tok = Tokenizer()
    t1 = time.perf_counter()
    encoded = encode_tok.encode(all_text)
    t2 = time.perf_counter()
    decode_tok = Tokenizer()
    t3 = time.perf_counter()
    decoded = decode_tok.decode(encoded)
    t4 = time.perf_counter()

    init_time = t1 - t0
    encode_time = t2 - t1
    decode_init_time = t3 - t2
    decode_time = t4 - t3
    total_time = t4 - t0

    print(f"  corpus: {total_chars:,} chars")
    print(f"  encoder init: {init_time:.3f}s")
    print(f"  encode:       {encode_time:.3f}s  ({total_chars/max(encode_time,1e-9):,.0f} chars/sec)")
    print(f"  decoder init: {decode_init_time:.3f}s")
    print(f"  decode:       {decode_time:.3f}s  ({total_chars/max(decode_time,1e-9):,.0f} chars/sec)")
    print(f"  total:        {total_time:.3f}s for {total_chars:,} chars")

    chars_per_sec = total_chars / total_time
    for label, n_chars, budget_sec in [
        ("debug corpus", 50_000, 600),
        ("final corpus", 2_000_000, 20 * 60),
    ]:
        projected = n_chars / chars_per_sec
        status = "OK" if projected < budget_sec * 0.5 else ("TIGHT" if projected < budget_sec else "FAIL")
        print(f"  [{status}] projected time for {label} ({n_chars:,} chars): "
              f"{projected:.1f}s (budget {budget_sec}s)")
    return decoded == all_text


def main():
    results = {
        "vocab_size": check_vocab_size(),
        "losslessness": check_losslessness(),
        "edge_cases": check_edge_cases(),
        "timing_and_roundtrip": check_timing(),
    }
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    if all(results.values()):
        print("\nAll checks passed.")
    else:
        print("\nSOME CHECKS FAILED -- see above.")


if __name__ == "__main__":
    main()
