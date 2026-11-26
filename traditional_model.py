"""ARTI-404 Group 5 — TFIDF: TF-IDF + Logistic Regression.

Modes:
  demo   : provisional AG News train/validation split; NEVER touches the official test.
  shared : final team train.csv/val.csv; optionally evaluate on test.csv after approval.
  predict: classify new headlines using a saved model, with no retraining.

Labels follow the official sh0416/ag_news dataset: 1 World, 2 Sports,
3 Business, 4 Sci/Tech. The system uses the headline from the text column (not description).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

LABEL_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
NAME_TO_LABEL = {
    "world": 1,
    "sports": 2,
    "business": 3,
    "sci/tech": 4,
    "science and technology": 4,
    "science & technology": 4,
    "science and tech": 4,
}
SEED = 42


def _normalized_title(title: str) -> str:
    """Same headline with differing whitespace/case is the same example."""
    return re.sub(r"\s+", " ", str(title).strip()).casefold()


def _detect_label_base(frames: list[pd.DataFrame], preferred: str = "auto") -> int:
    if preferred in {"0", "1"}:
        return int(preferred)
    values = pd.concat([f["label"] for f in frames], ignore_index=True)
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return 1  # named class labels
    if (nums == 0).any() and (nums == 4).any():
        raise ValueError("Mixed label bases (0 and 4) detected. Use consistent labels across splits.")
    if (nums == 0).any():
        return 0
    if (nums == 4).any():
        return 1
    raise ValueError("Cannot infer label base from classes 1..3 alone. Pass --label-base 0 or 1.")


def _canonical_frame(df: pd.DataFrame, label_base: int) -> pd.DataFrame:
    if not {"label"}.issubset(df.columns) or not ({"title", "text"} & set(df.columns)):
        raise ValueError("Expected 'label' plus either 'text' (Ibrahim) or 'title' (raw data).")
    df = df.copy()
    if 'text' in df.columns:
        # Ibrahim already prepared the official headline-only column.
        df['title'] = df['text']
    else:
        df['text'] = df['title']
    if df[["title", "label"]].isna().any().any():
        raise ValueError("Missing title or label. Ask Ibrahim to clean the shared split.")
    df["title"] = df["title"].astype(str).str.strip()
    if (df["title"].str.len() == 0).any():
        raise ValueError("Empty title in shared data. Ask Ibrahim to clean it.")

    def convert(value):
        if isinstance(value, str) and value.strip().casefold() in NAME_TO_LABEL:
            return NAME_TO_LABEL[value.strip().casefold()]
        try:
            number = int(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unrecognized class label: {value!r}") from exc
        if number != float(value):
            raise ValueError(f"Non-integer class label: {value!r}")
        number += (1 if label_base == 0 else 0)
        if number not in LABEL_NAMES:
            raise ValueError(f"Class outside four AG News classes: {value!r}")
        return number

    df["label"] = df["label"].map(convert).astype(int)
    return df.reset_index(drop=True)


def _check_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame | None = None) -> None:
    sets = {
        "train": set(train["title"].map(_normalized_title)),
        "val": set(val["title"].map(_normalized_title)),
    }
    if test is not None:
        sets["test"] = set(test["title"].map(_normalized_title))
    keys = list(sets)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            common = sets[left] & sets[right]
            if common:
                raise ValueError(
                    f"{len(common)} duplicate headline(s) overlap {left} and {right}. "
                    "Ibrahim must fix the split before evaluation. "
                    f"Example: {next(iter(common))!r}"
                )


def load_demo_data(sample_size: int = 8000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporary development split: official AG News train only; no test access."""
    if sample_size < 100:
        raise ValueError("Choose --sample-size >= 100 so all four classes are represented.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install dependencies: python -m pip install -r requirements.txt") from exc
    raw = load_dataset("sh0416/ag_news", split="train")
    if sample_size > len(raw):
        sample_size = len(raw)
    # Shuffle before subsetting; classes in the original dataset are ordered.
    raw = raw.shuffle(seed=SEED).select(range(sample_size))
    frame = pd.DataFrame({"title": raw["title"], "label": raw["label"]})
    frame = _canonical_frame(frame, label_base=1)
    frame["_dedup_key"] = frame["title"].map(_normalized_title)
    n_before = len(frame)
    frame = frame.drop_duplicates("_dedup_key", keep="first").drop(columns="_dedup_key")
    if n_before != len(frame):
        print(f"[Demo] Removed {n_before - len(frame)} duplicated headlines before splitting.")
    train, val = train_test_split(frame, test_size=0.2, stratify=frame["label"], random_state=SEED)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    _check_no_overlap(train, val)
    return train, val


def load_shared_data(
    data_dir: str | Path, *, include_test: bool = False, label_base: str = "auto"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load Ibrahim's agreed splits without changing the row order."""
    data_dir = Path(data_dir)
    requested = ["train.csv", "val.csv"] + (["test.csv"] if include_test else [])
    for name in requested:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing {data_dir / name} — request this file from Ibrahim.")
    raw_frames = [pd.read_csv(data_dir / name) for name in requested]
    base = _detect_label_base(raw_frames, label_base)
    frames = [_canonical_frame(frame, base) for frame in raw_frames]
    train, val = frames[0], frames[1]
    test = frames[2] if include_test else None
    _check_no_overlap(train, val, test)
    if set(train["label"]) != set(LABEL_NAMES):
        raise ValueError("Training split must include all four AG News labels.")
    return train, val, test


def _make_pipeline(ngram_range: tuple[int, int], c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                    max_features=40_000,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", OneVsRestClassifier(LogisticRegression(
                C=c_value, solver="liblinear", max_iter=300, tol=1e-3, random_state=SEED
            ))),
        ]
    )


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[1, 2, 3, 4], average="macro", zero_division=0)),
    }


def predict_frame(model: Pipeline, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Preserve the input row order so teammate predictions can be compared by row_id."""
    predictions = model.predict(frame["title"])
    out = pd.DataFrame({
        "row_id": range(len(frame)),
        "title": frame["title"].values,
        "text": frame["text"].values,
        "prediction": predictions.astype(int),
        "true_label": frame["label"].map(LABEL_NAMES).values,
        "predicted_label": pd.Series(predictions).map(LABEL_NAMES).values,
    })
    out["correct"] = out["true_label"] == out["predicted_label"]
    out["word_count"] = out["title"].str.split().map(len)
    out["short_6_or_less"] = out["word_count"] <= 6

    result = {"overall": _metrics(frame["label"], predictions), "samples": int(len(frame))}
    short_mask = out["short_6_or_less"].to_numpy()
    if short_mask.any():
        result["short_6_or_less"] = _metrics(frame["label"][short_mask], predictions[short_mask])
        result["short_samples"] = int(short_mask.sum())
    else:
        result["short_6_or_less"] = None
        result["short_samples"] = 0
    return out, result


def train_and_select(
    train: pd.DataFrame, val: pd.DataFrame, *, quick: bool = False
) -> tuple[Pipeline, list[dict], pd.DataFrame, dict, dict]:
    """Select settings by validation MACRO F1; train only on TRAIN."""
    if quick:
        candidates = [((1, 1), 1.0), ((1, 2), 1.0)]
    else:
        candidates = [(n, c) for n in [(1, 1), (1, 2)] for c in [0.5, 2.0, 8.0]]
    leaderboard: list[dict] = []
    best_score = (-1.0, -1.0)
    best_model = None
    best_settings = {}
    for i, (ngrams, c_value) in enumerate(candidates, 1):
        print(f"Training {i}/{len(candidates)} — ngram_range={ngrams}, C={c_value}")
        model = _make_pipeline(ngrams, c_value)
        model.fit(train["title"], train["label"])
        predicted = model.predict(val["title"])
        scores = _metrics(val["label"], predicted)
        leaderboard.append({"ngram_range": str(ngrams), "C": c_value, **scores})
        ranking = (scores["macro_f1"], scores["accuracy"])
        if ranking > best_score:
            best_score = ranking
            best_model = model
            best_settings = {"ngram_range": list(ngrams), "C": c_value}
    assert best_model is not None
    val_predictions, val_summary = predict_frame(best_model, val)
    return best_model, leaderboard, val_predictions, val_summary, best_settings


def save_outputs(
    output_dir: str | Path,
    model: Pipeline,
    leaderboard: list[dict],
    val_predictions: pd.DataFrame,
    val_summary: dict,
    best_settings: dict,
    *, mode: str,
) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "tfidf_logreg.joblib"
    joblib.dump(model, model_path)
    pd.DataFrame(leaderboard).sort_values("macro_f1", ascending=False).to_csv(folder / "hyperparameter_comparison.csv", index=False)
    val_predictions.to_csv(folder / "validation_predictions.csv", index=False)
    val_predictions.loc[~val_predictions["correct"]].head(30).to_csv(folder / "validation_errors.csv", index=False)
    summary = {
        "data_mode": mode,
        "model": "TF-IDF + Logistic Regression",
        "input_column": "text",
        "label_mapping": LABEL_NAMES,
        "validation_metrics": val_summary,
        "selected_settings": best_settings,
        "random_seed": SEED,
        "test_used_for_selection": False,
    }
    (folder / "model_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model_path


def predict_text(model: Pipeline, title: str) -> str:
    if not isinstance(title, str) or not title.strip():
        return "Please enter a non-empty English news headline."
    return LABEL_NAMES[int(model.predict([title.strip()])[0])]



def export_frozen_test(
    *, data_dir: str | Path = "data",
    model_path: str | Path = "outputs/tfidf/tfidf_logreg.joblib",
    output_dir: str | Path = "outputs/tfidf",
    results_dir: str | Path = "results",
) -> dict:
    """Evaluate the previously selected frozen model exactly once for the final report.

    Does NOT fit or choose any settings based on the test split. Preserves
    test row order and exports the file expected by the team's evaluator.
    """
    train, val, test = load_shared_data(data_dir, include_test=True)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}. Run training first.")
    model = joblib.load(model_path)  # Load only models from trusted project sources.
    model_info_path = model_path.with_name("model_info.json")
    if model_info_path.is_file():
        info = json.loads(model_info_path.read_text(encoding="utf-8"))
        settings = info.get("selected_settings", {})
        tfidf = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        if settings:
            current = (tuple(tfidf.ngram_range), float(classifier.estimators_[0].C))
            expected = (tuple(settings["ngram_range"]), float(settings["C"]))
            if current != expected:
                raise ValueError(f"Model parameters do not match chosen validation settings: {current} vs {expected}")

    # Sanity-check frozen model on validation (not used to change settings).
    _, validation_metrics = predict_frame(model, val)
    preds, test_metrics = predict_frame(model, test)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = Path(results_dir)
    result.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "test_predictions_detailed.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (out / "frozen_validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    # Required team interface: exactly the same rows/order/text as data/test.csv.
    team_file = result / "tfidf_predictions.csv"
    preds[["text", "prediction"]].to_csv(team_file, index=False)

    class_ids = [1, 2, 3, 4]
    cm = pd.DataFrame(
        confusion_matrix(test["label"], preds["prediction"], labels=class_ids),
        index=[f"actual_{LABEL_NAMES[i]}" for i in class_ids],
        columns=[f"predicted_{LABEL_NAMES[i]}" for i in class_ids],
    )
    cm.to_csv(result / "tfidf_confusion_matrix.csv")
    (result / "tfidf_classification_report.txt").write_text(
        classification_report(test["label"], preds["prediction"], labels=class_ids,
                              target_names=[LABEL_NAMES[i] for i in class_ids], zero_division=0),
        encoding="utf-8")
    preds.loc[~preds["correct"]].head(30).to_csv(result / "tfidf_error_examples.csv", index=False)
    (result / "tfidf_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Frozen model: {model_path}")
    print(f"Validated {len(train):,} training rows, {len(val):,} validation rows, {len(test):,} test rows")
    print("TEST (one-time evaluation; do not retune):", json.dumps(test_metrics, indent=2))
    print(f"Team prediction file: {team_file}")
    return test_metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="Group 5: TF-IDF + Logistic Regression")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Use a temporary stratified split of AG News training examples")
    demo.add_argument("--sample-size", type=int, default=8000)
    demo.add_argument("--output-dir", default="outputs/tfidf_demo")
    demo.add_argument("--quick", action="store_true", help="Try two configurations instead of six")

    shared = sub.add_parser("shared", help="Use Ibrahim's train.csv and val.csv")
    shared.add_argument("--data-dir", default="data")
    shared.add_argument("--output-dir", default="outputs/tfidf")
    shared.add_argument("--label-base", choices=["auto", "0", "1"], default="auto")
    shared.add_argument("--quick", action="store_true")
    shared.add_argument("--final-eval", action="store_true", help="ONLY after validation settings are frozen: evaluate held-out test.csv once")

    frozen = sub.add_parser("export-test", help="Export test predictions from frozen trained model; no retraining")
    frozen.add_argument("--data-dir", default="data")
    frozen.add_argument("--model-path", default="outputs/tfidf/tfidf_logreg.joblib")
    frozen.add_argument("--output-dir", default="outputs/tfidf")
    frozen.add_argument("--results-dir", default="results")

    predict = sub.add_parser("predict", help="Predict new headline without retraining")
    predict.add_argument("--model-path", default="outputs/tfidf/tfidf_logreg.joblib")
    predict.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "export-test":
        export_frozen_test(data_dir=args.data_dir, model_path=args.model_path,
                           output_dir=args.output_dir, results_dir=args.results_dir)
        return
    if args.command == "predict":
        if not Path(args.model_path).is_file():
            parser.error("Model not found. Train using demo/shared first and provide --model-path.")
        model = joblib.load(args.model_path)
        print(predict_text(model, args.text))
        return

    if args.command == "demo":
        train, val = load_demo_data(args.sample_size)
        test = None
        print("Demo mode: AG News training data ONLY. Official test is never loaded.")
    else:
        train, val, test = load_shared_data(args.data_dir, include_test=args.final_eval, label_base=args.label_base)
    print(f"Train: {len(train):,} | Validation: {len(val):,}")
    model, leaderboard, pred, summary, settings = train_and_select(train, val, quick=args.quick)
    model_path = save_outputs(args.output_dir, model, leaderboard, pred, summary, settings, mode=args.command)
    print("\nBest settings:", settings)
    print("Validation metrics:", json.dumps(summary, indent=2))
    print("Saved model:", model_path)
    print("Try it example:", predict_text(model, "Team wins football final"))

    if args.command == "shared" and args.final_eval:
        assert test is not None
        test_pred, final_metrics = predict_frame(model, test)
        folder = Path(args.output_dir)
        test_pred.to_csv(folder / "test_predictions.csv", index=False)
        (folder / "test_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        # Exact schema expected by Ibrahim's evaluation_legacy.py. No shuffling.
        team_results = Path("results")
        team_results.mkdir(parents=True, exist_ok=True)
        team_file = team_results / "tfidf_predictions.csv"
        test_pred[["text", "prediction"]].to_csv(team_file, index=False)
        print("Ibrahim compatible predictions:", team_file)
        print("\nFINAL test metrics (do not use to tune settings):", json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    _main()
