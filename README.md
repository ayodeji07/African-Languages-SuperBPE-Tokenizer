# SuperBPE Tokenizer for African LLMs — Team Kyoga

A from-scratch SuperBPE tokenizer (Liu et al., 2025) for the "Bridging the Tokenization Gap in African LLMs" challenge, under a 20,000-token vocab cap. Final submission: 367,839 tokens on the 2M-char held-out corpus, 5.64s runtime.

## Dataset

**13 training languages**: Amharic, Hausa, Somali, Swahili, Yoruba, Igbo, Kinyarwanda, isiXhosa, Nigerian Pidgin, Luo, Kanuri, Dinka, Nama.
**4 held-out generalization-test languages** (never trained on): Tigrinya, Zulu, Ewe, Nuer — each mirrors a trained language's family/script to test whether the tokenizer generalizes rather than memorizes.

Sources: [WURA](https://huggingface.co/datasets/castorini/wura) for 8 higher-resource languages, eBible.org for Dinka and Kanuri, Glot500 for Luo, small verified-clean sources for Nigerian Pidgin and Nama. Mined/NLLB-style sources showed 20-70%+ cross-lingual contamination, so two independent filters — script-based (`filter_script_contamination.py`) and fastText-langid-based (`filter_langid_contamination.py`) — were built before using any bitext-derived source.

**Mixing**: languages are blended by temperature sampling, `share_i ∝ volume_i^alpha`, with **alpha=0.10** found via a sweep minimizing worst-case per-language compression (not just the average). Re-confirmed after adding SuperBPE's second stage and again after adding isiXhosa.

isiXhosa was promoted from a Tier-2 test language into training after its generalization score revealed a large gap; validated with a real score improvement (10,044 → 9,618). Zulu was tested the same way — safe at small scale, but a persistent 16-25% Yoruba regression appeared at larger scale, confirmed as a net loss on the real score (9,649 vs 9,618) and rejected.

## Training Pipeline

SuperBPE has no existing library implementation, so the trainer (`scripts/training/bpe_trainer.py`) was built from scratch: an O(n log n) max-heap BPE learner with lazy staleness checking, scoped updates, and periodic heap rebuilds to bound memory.

**Two-stage training** (`train_superbpe_pilot.py`): Stage 1 learns ordinary word-internal merges up to vocab 8,000. Stage 2 removes whitespace pretokenization and learns "superword" merges that cross word boundaries, up to the 20,000-token cap. This matters specifically because non-ASCII characters are preprocessed into readable ASCII markers (`[U+XXXX NAME]`) that contain internal spaces — efficiently compressing marker-heavy languages (Amharic: 23x character expansion; Dinka/Nama: ~7x; Igbo: 5x; Yoruba: 4.8x) structurally requires whitespace-crossing merges.

Training corpus: 100M characters (reduced from an originally-planned 336M after memory constraints on the training hardware; validated this doesn't hurt quality — held-out compression barely changed between a 14M-character pilot and the 100M-character run, confirming diminishing returns to raw data volume well before 336M).

**Alternatives tested and rejected**: the SuperBPE paper's own max-merge-span cap (hurt compression — trades it for downstream-task robustness, not our objective); a lower stage-1/2 transition point (no gain); WordPiece-style PMI merge scoring (dramatically worse everywhere, since it optimizes model quality, not raw token count).

Checkpointing saves full trainer state, not just the merge list, so an interrupted run resumes in seconds instead of re-deriving state from scratch.

## Evaluation

- **Losslessness**: `decode(encode(x)) == x` verified across every held-out line for all 13 languages plus adversarial edge cases (empty string, tabs, all-punctuation, repeated tokens), using two independent `Tokenizer()` instances to match the real evaluator's usage exactly.
- **Training curve**: tokens/char tracked on training and held-out samples across vocab-size snapshots (`collect_training_curve.py`) — both track closely (no overfitting) and are still declining at the 20,000-token cap (vocab budget, not data volume, is the binding constraint).
- **Tier-2 generalization**: measured compression and whitespace-crossing usage on 4 never-trained-on languages to catch overfitting to the training set.
- **Real-platform validation**: every candidate was submitted to Codabench's dev phase before committing the final one — pilot (10,145) → full-scale (10,044) → +isiXhosa (9,618, best) → +Zulu (9,649, rejected) → alpha re-tune (9,654, no gain). Internal held-out metrics did not reliably predict the real score, so no candidate was trusted without one.

## Reproduction

**Quick test** (seconds, stdlib only) — from `scripts/final_submission/`, the exact code/vocab from the final submission:
```python
from tokenizer import Tokenizer
ids = Tokenizer().encode(["your text here"])
print(Tokenizer().decode(ids))
```

**Full pipeline** (hours, needs real compute/bandwidth) — run in order from `scripts/`:

1. `data_pipeline/fetch_wura_samples.py`, `fetch_xhosa_zulu.py` — download raw per-language text.
2. `data_pipeline/filter_script_contamination.py`, `filter_langid_contamination.py` — remove contamination from mined sources.
3. `data_pipeline/compute_mixing_ratios.py` — compute alpha-weighted per-language target sizes.
4. `data_pipeline/assemble_pilot_corpus.py --alpha 0.10 --total-processed-chars 100000000` — build the final mixed, preprocessed corpus and its held-out split.
5. `training/train_superbpe_pilot.py <corpus> <out.json> --stage1-vocab-size 8000 --vocab-size 20000` — train the two-stage tokenizer.
6. `evaluation/validate_submission.py`, `evaluation/evaluate_tier2.py` — verify losslessness, timing, generalization.

`data/raw_samples/` holds small per-language samples for quick inspection; full training/held-out data is regenerated by the pipeline above, not checked in (multiple gigabytes across sources).

## Appendix

**Contributors**: _[Ayodeji Akande, Michael Ajiboye, Hezekiah Joel Abiodun, Joy Idoko, Anya Happiness Anuri, Oladipo Solomon Olamidokun]_<br>
**Mentor**: _[Elinah Moyo]_
