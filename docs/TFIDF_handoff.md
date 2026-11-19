# TFIDF handoff to Ibrahim

The TF-IDF model and its test predictions are complete. **No pretrained-model comparison is included**; that analysis must wait for Mohammed's prediction file.

## Required team file

- `results/tfidf_predictions.csv`: two columns `text,prediction`, matching the supplied `data/test.csv` exactly in original row order, with class IDs 1–4.

Copy this file into the **group repository** at the same relative path. Ibrahim's `evaluation_member1.py` checks the row ordering and compares this file with `results/minilm_predictions.csv` once that is available.

## Model code and files

- `traditional_model.py` — reproducible feature extraction, training/validation selection, prediction and frozen-model test export.
- `TFIDF.ipynb` — readable and executable notebook covering setup, data, method, validation, results, Try it, sources.
- `outputs/tfidf/tfidf_logreg.joblib` — saved fitted model and vectorizer; trusted project file.
- `outputs/tfidf/hyperparameter_comparison.csv` — six validation runs.
- `results/tfidf_test_metrics.json`, `results/tfidf_classification_report.txt`, `results/tfidf_confusion_matrix.csv`, `results/tfidf_error_examples.csv` — standalone TF-IDF test analysis.
- `docs/TFIDF_report_section.md` — results and a report subsection draft.

## Rules for team integration

1. Use the same prepared `train.csv`, `val.csv`, `test.csv` and `text` field for both methods.
2. Do not shuffle predictions; `text,prediction` must align with `test.csv` row for row.
3. Do not retune model parameters after viewing test results.
4. Compare Accuracy and Macro F1 on the full test set and on the `is_short` subset.
5. Review at least three errors and document individual contributions and two **personally created** labelled examples per student.
6. Keep the final Notebook, Report, Slides and Appendix consistent and credit assistance/tools.

## Re-run the saved frozen model

```
python traditional_model.py export-test --data-dir data --model-path outputs/tfidf/tfidf_logreg.joblib
```

No comparison is possible until `results/minilm_predictions.csv` is provided by Mohammed.
