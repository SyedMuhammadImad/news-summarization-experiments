# News summarization

BART encoder-decoder generation with truncation and ROUGE-1/2/L scoring. No fine-tuning or unsupported training claim.

The earlier notebook was repaired into a reusable module plus the same setup → experiment → results notebook flow. Original source files remain on the laptop. See `VERIFICATION.json` for completed checks and `metrics.json` for fresh results when available.

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python summarize.py --data test.csv --limit 10 --output metrics.json
```

Supply your dataset files at the indicated paths. Raw corpora, model binaries, credentials, pictures and videos are excluded. Pretrained model downloads happen locally. Evaluation sizes and dataset limitations are explicit in the results; small runs are functional evidence, not a broad benchmark.
