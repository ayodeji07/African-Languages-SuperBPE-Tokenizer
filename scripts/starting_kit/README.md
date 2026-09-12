# Starting kit

Welcome to **Build a SuperBPE Tokenizer**.

The included baseline is deliberately simple and lossless. It maps each Unicode
character to an integer, so it demonstrates the required interface but does not
provide useful compression.

The evaluation corpus is preprocessed before your tokenizer sees it. Inputs are
ASCII-only strings: whitespace is collapsed to single spaces, punctuation is
simplified, accented and non-Latin source characters are represented with ASCII
transliterations plus explicit `[U+.... NAME]` markers.

See `preporocess_text.py` for the reference preprocessing function used to
produce the public and evaluation corpora.

Your tokenizer should handle the full ASCII symbol set, including punctuation,
quotes, brackets, digits, letters, and spaces.

This challenge expects a symbol-level tokenizer over preprocessed ASCII
strings, not a raw UTF-8 byte-level tokenizer. Unlike many production
tokenizers that use byte-level BPE or byte fallback, your tokenizer receives
ASCII `str` inputs where non-ASCII source characters have already been written
as ASCII markers.

The final corpus contains 2,000,000 post-processed characters, and the full
encode/decode evaluation must finish within 20 minutes on the competition
platform. During development, benchmark on a large corpus and estimate your
runtime for the final platform workload.

The platform uses the `codalab/codalab-legacy:py312` Docker image with Python
3.12. Installed packages include `Cython==3.0.12`, `numpy==1.26.4`,
`scipy==1.11.4`, `scikit-learn==1.5.1`, `pandas==2.2.3`, `pyyaml==6.0.2`,
`imutils==0.5.4`, `numba==0.61.2`, `threadpoolctl==3.6.0`,
`matplotlib==3.8.4`, and `psutil==7.0.0`. Do not rely on network access during
evaluation.

You have already learned how to implement ordinary BPE. Replace the baseline
with your trained SuperBPE tokenizer, allowing later merges to cross word
boundaries while preserving enough information for exact decoding.

Before implementing your tokenizer, read
[SuperBPE: Space Travel for Language Models](https://arxiv.org/pdf/2503.13423).
Use the paper as inspiration for a tokenizer that begins with BPE-style merging
and then learns useful tokens that may cross whitespace, while keeping decoding
lossless.

Files:

- `tokenizer.py`: minimal valid baseline and required API
- `preporocess_text.py`: reference text preprocessing helper
- `sample-code-submission.zip`: ready-to-upload baseline submission

Your final ZIP may also include a static `tokenizer.json` containing learned
vocabulary and merge rules. Keep `tokenizer.py` at the archive root.
