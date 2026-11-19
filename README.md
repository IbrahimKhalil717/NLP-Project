# News Topic Classification — Group 5 (ARTI-404)

A complete, runnable **headline-only** NLP comparison for AG News. Research question: *Which method performs better on very short headlines (six words or fewer): TF-IDF with Logistic Regression or pretrained all-MiniLM-L6-v2 sentence embeddings with Logistic Regression?*

**Two frozen models:**
- **TF-IDF + one-vs-rest Logistic Regression** — saved pipeline `outputs/tfidf/tfidf_logreg.joblib`; best validated unigram TF-IDF, `C=8.0`.
- **Pretrained all-MiniLM-L6-v2 sentence representations + Logistic Regression** — `models/minilm_logreg_coefficients.npz` (portable prediction weights) and `models/embedder_name.txt`; best validated `C=0.5`.

Labels: **1 World, 2 Sports, 3 Business, 4 Sci/Tech**. Internally the MiniLM classifier uses labels 0–3; its canonical exported predictions use 1–4.

## Quick start

Use **Python 3.11** or newer and an isolated environment. `requirements.txt` specifies the tested libraries (including `scikit-learn==1.8.0` needed for the saved TF-IDF model). **Never load unknown untrusted `.joblib` files.**

```bash
python -m pip install -r requirements.txt
jupyter notebook News_Topic_Classification.ipynb
```

Open Jupyter **from the project root**, then **Kernel → Restart & Run All**. The submitted main notebook includes setup, data, Method 1, Method 2, comparison, an analysis of three real errors, **Try it**, and sources. Saved cell outputs are preserved.

The `data/*.csv` files are provided in the complete ZIP. Git may omit them because the repository `.gitignore` intentionally excludes generated datasets. If the data files are absent, regenerate them from the public [AG News dataset](https://huggingface.co/datasets/sh0416/ag_news):

```bash
python preprocessing.py
```

This downloads original AG News source files and reconstructs the same deterministic train/val/test procedure. Dataset updates upstream may affect exact byte-for-byte reproducibility; supplied split manifests can verify the approved files.

## Reproduce final comparison (no retraining)

The original MiniLM predictions are stored in `results/predictions_B.csv` using label IDs **0..3** and lightly cleaned headline strings. The first command validates all **7,464** source rows against the original test labels and cleaned text, then exports `results/minilm_predictions.csv` with unchanged predictions and **1..4** label IDs. It restores original text exactly in test order. Do **not** simply rename the source file.

```bash
python prepare_minilm_predictions.py
python evaluation.py
```

The evaluator enforces identical row order/text, correct labels and short-headline flags; writes `results/metrics.csv`, confusion matrices, classification reports, `results/errors_to_analyze.csv`, `results/short_headline_errors.csv`, and `results/comparison_summary.json`.

| Test subset | Method | Accuracy | Macro F1 | n |
|---|---|---:|---:|---:|
| All headlines | TF-IDF | 88.29% | 88.23% | 7,464 |
| All headlines | MiniLM | 83.82% | 83.75% | 7,464 |
| Very short (≤6 words) | TF-IDF | 86.72% | 86.56% | 3,554 |
| Very short (≤6 words) | MiniLM | 81.06% | 80.95% | 3,554 |

See `docs/error_analysis.md` for three explained test mismatches and limitations.

## Try it — New headlines

TF-IDF works offline with the saved pipeline:

```bash
python predict_news.py --model tfidf --text "Team wins football final"
```

MiniLM live inference downloads pretrained weights on **first** use and may require Internet access:

```bash
python predict_news.py --model minilm --text "Scientists discover a new way to charge batteries"
```

Both methods reject empty text with a helpful message and do not retrain while predicting. MiniLM uses the exported multinomial logistic-regression NumPy coefficient arrays instead of loading the old classifier pickle (trained in scikit-learn 1.9.1, which differs from the TF-IDF version). The existing `models/pretrained_lr.joblib` is retained only as a historical training artifact, not required for live predictions.

## Training source and reproducibility

Training code is included for inspection and independent reproduction:

```bash
python traditional_model.py shared --data-dir data
python pretrained_model.py --data_dir data --out_dir results --model_dir models
```

**Do not run training again to reproduce submitted scores** without preserving the frozen model files first. The trained versions and results are in the archive. The MiniLM training script uses pretrained weights `sentence-transformers/all-MiniLM-L6-v2` and trains the classifier on train data, tuning on validation. Generating embeddings for all splits is computationally expensive and downloads a large pretrained model. `pretrained_model.py` has corrected default paths and now also outputs the canonical prediction format and coefficient file when retrained.

## Files

- `News_Topic_Classification.ipynb` — **primary** team notebook to submit; execute `Run All` from clean kernel.
- `TFIDF.ipynb` — optional original development notebook; **not** the main submission.
- `preprocessing.py`, `traditional_model.py`, `pretrained_model.py` — source methods.
- `prepare_minilm_predictions.py`, `evaluation.py`, `predict_news.py` — canonical integration, evaluation and inference.
- `evaluation_legacy.py` — compatibility command that delegates to `evaluation.py`.
- `data/*.csv` — frozen approved splits; `data/preprocessing_stats.json` gives counts.
- `results/*.csv` — final fixed model predictions, per-class reports and error-case evidence.
- `outputs/tfidf/` and `models/` — saved models and evidence.
- `docs/` — technical report sections, contribution evidence, documented error analysis and two original TF-IDF examples for personal review.
- `deliverables/` — **draft** joint report/presentation/appendix, to be finalized with all students.

## Academic integrity and attributions

Dataset: [AG News](https://huggingface.co/datasets/sh0416/ag_news). Pretrained weights: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Libraries: scikit-learn, pandas, Sentence Transformers. Technical references: Zhang et al. (2015), *Character-level Convolutional Networks for Text Classification* [paper](https://arxiv.org/abs/1509.01626); Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* [paper](https://aclanthology.org/D19-1410/).

AI assistance was used for code support, debugging, documentation and drafting. Each student must review and personally label their **two original examples** and confirm their own documented contribution. Original examples must be **separate** from the test metrics. The team must supply an appendix, an 8-page-or-less seven-section report and a 5–10-slide presentation (about 12 minutes, every student speaks). The course's final ZIP is named `GroupID_Project.zip` and is uploaded to Blackboard, with a GitHub link.
