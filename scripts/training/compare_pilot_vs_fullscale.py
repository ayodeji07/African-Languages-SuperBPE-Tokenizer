"""One-off: measure the pilot tokenizer (submission/tokenizer.json) on the
same held-out set used for the full-scale training curve, so the two numbers
are directly comparable (same measurement code, same held-out data).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "eval_root/submission")
from tokenizer import Tokenizer  # noqa: E402

pilot_data = json.loads(Path("eval_root/pilot_tokenizer.json").read_text(encoding="utf-8"))
# swap in the pilot's vocab by writing it to the path tokenizer.py expects,
# then restore the full-scale one after
tok_json_path = Path("eval_root/submission/tokenizer.json")
fullscale_data = tok_json_path.read_text(encoding="utf-8")
tok_json_path.write_text(json.dumps(pilot_data), encoding="utf-8")

tok = Tokenizer()
total_tokens = total_chars = 0
per_lang = {}
for p in sorted(Path("eval_root/submission_test_data").glob("*_heldout_processed.txt")):
    lang = p.stem.replace("_heldout_processed", "")
    lines = [ln for ln in p.read_text(encoding="utf-8").split("\n") if ln]
    encoded = tok.encode(lines)
    n_tok = sum(len(e) for e in encoded)
    n_chr = sum(len(l) for l in lines)
    per_lang[lang] = n_tok / n_chr
    total_tokens += n_tok
    total_chars += n_chr

print("PILOT overall tokens/char:", total_tokens / total_chars)
for k, v in sorted(per_lang.items()):
    print(f"  {k}: {v:.4f}")

tok_json_path.write_text(fullscale_data, encoding="utf-8")
print("(restored full-scale tokenizer.json)")
