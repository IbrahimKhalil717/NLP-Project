# Archived TFIDF handoff notes (before team integration)

**Superseded by the main README.md.** These original notes predate the team integration and must not be followed for final submission.


Unzip **into the root** of the existing `NLP-Project` repository. Do not replace Ibrahim's `README.md`, `requirements.txt`, `preprocessing.py` or `evaluation_member1.py`.

## Installation

Use Python 3.11+ and install the model-specific dependencies:

```bash
python -m pip install -r requirements_tfidf.txt
```

This includes scikit-learn 1.8.0 to load the saved model safely. For a fresh project setup, also install the team's general requirements as needed.

## Files to commit

- `TFIDF.ipynb`, `traditional_model.py`, `README_TFIDF.md`, `requirements_tfidf.txt`
- `outputs/tfidf/` (model `tfidf_logreg.joblib`, tuning table and validation/test evidence)
- `results/tfidf_predictions.csv` **(required by Ibrahim's evaluation script)**
- `results/tfidf_test_metrics.json`, `results/tfidf_confusion_matrix.csv`, `results/tfidf_classification_report.txt`, `results/tfidf_error_examples.csv`
- `docs/` (handoff, report draft and personal-example template)

The group repository's **existing `.gitignore` ignores `*.joblib`, `results/*.csv`, and `results/*.txt`**. Therefore, Git will omit the model and required prediction file unless you explicitly whitelist or force-add them. Add these override lines **at the end** of the existing `.gitignore` (recommended):

```
!outputs/tfidf/tfidf_logreg.joblib
!results/tfidf_predictions.csv
!results/tfidf_confusion_matrix.csv
!results/tfidf_error_examples.csv
!results/tfidf_classification_report.txt
```

Or, for only the essential handoff files, run:

```bash
git add -f outputs/tfidf/tfidf_logreg.joblib results/tfidf_predictions.csv
```

Check with `git status` that those files really appear as staged. Do not commit the shared `data/*.csv` files: they are created by Ibrahim's `preprocessing.py` and must match his team's exact split. His project must already have `data/train.csv`, `data/val.csv` and `data/test.csv` available locally when executing the notebook.

## Ready for Ibrahim's evaluation

After Mohammed provides `results/minilm_predictions.csv`, Ibrahim can run:

```
python evaluation_member1.py
```

Ibrahim's evaluator was checked successfully against our `results/tfidf_predictions.csv` (row order, column names, class values).

If you need an offline, self-contained copy with the exact data, use the other archive `TFIDF_Complete.zip` instead.
