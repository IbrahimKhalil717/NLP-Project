"""
pretrained_model.py
===================
Member 3  : Mohammed Alshahrani   (ID: 2230007324)
Course    : ARTI-404 - Group 5
Project   : News Topic Classification for Short News Headlines (Option 3)

Role of this file
-----------------
Builds and tests the PRETRAINED EMBEDDING model (Method B of the proposal):

    all-MiniLM-L6-v2  -->  Sentence Embedding (384-d)  -->  Logistic Regression  -->  {World, Sports, Business, Sci/Tech}

Pipeline position
-----------------
Member 1 prepares data (train.csv / val.csv / test.csv with columns:
    text, label, word_count, is_short)
Member 2 trains the TF-IDF + LR model (traditional_model.py)
Member 3 (this file) trains the Sentence-Transformer + LR model
Member 1 then evaluates both and runs error analysis.

How to run
----------
    python pretrained_model.py \
        --data_dir  data \
        --out_dir   results \
        --model_dir models

The script writes:
    models/pretrained_lr.joblib          (fitted LogisticRegression on 384-d embeddings)
    models/embedder_name.txt             (the Sentence-Transformer identifier used)
    results/predictions_B.csv            (source predictions, 0..3 labels)
    results/minilm_predictions.csv       (team-compatible predictions, 1..4 labels)
    results/predictions_B_val.csv        (validation predictions)
    results/metrics_B.json               (val + test metrics, full and short-only)
    results/try_it_examples_B.csv        (draft Try-It examples requiring personal review)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

# -------------------------------------------------------------------------
# Constants fixed by the project
# -------------------------------------------------------------------------

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NAME_TO_ID  = {v: k for k, v in LABEL_NAMES.items()}

# AG News (sh0416/ag_news on HuggingFace) uses labels 1..4.
# Member 1's preprocessing keeps that same scheme, so index 0 is never used.
# We convert to 0..3 internally so sklearn is happy.
#
# After Member 1's preprocessing we see labels {1, 2, 3, 4} -> we map to {0, 1, 2, 3}.
LABEL_SHIFT = 1

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE   = 42


# -------------------------------------------------------------------------
# Cleaning (very light — the heavy work is done by Member 1)
# -------------------------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_WS       = re.compile(r"\s+")

def light_clean(text: str) -> str:
    """Drop stray HTML tags like <b>...</b> and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    t = _HTML_TAG.sub(" ", text)
    t = t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    t = _WS.sub(" ", t).strip()
    return t


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    df = pd.read_csv(path)
    required = {"text", "label", "word_count", "is_short"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["text"] = df["text"].astype(str).map(light_clean)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Shift labels 1..4 -> 0..3
    df["y"] = df["label"].astype(int) - LABEL_SHIFT
    if not set(df["y"].unique()).issubset({0, 1, 2, 3}):
        raise ValueError(
            f"Unexpected labels in {name}: {sorted(df['y'].unique())}"
        )
    return df


# -------------------------------------------------------------------------
# Embedding
# -------------------------------------------------------------------------
def build_embedder():
    """
    Load the all-MiniLM-L6-v2 sentence transformer.
    Kept in its own function so the import failure message is helpful.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "The 'sentence-transformers' package is required for Member 3's "
            "model. Install it with:\n    pip install sentence-transformers\n"
            f"Original error: {e}"
        )
    print(f"[embedder] loading {EMBED_MODEL_ID} ...")
    model = SentenceTransformer(EMBED_MODEL_ID)
    return model


def encode(model, texts, batch_size=64):
    """Return an (N, 384) float32 numpy array; embeddings are L2-normalised."""
    t0 = time.time()
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine-friendly
    )
    dt = time.time() - t0
    print(f"[embedder] encoded {len(texts):,} texts "
          f"-> {emb.shape} in {dt:.1f}s "
          f"({len(texts)/max(dt, 1e-6):.0f} texts/s)")
    return emb.astype(np.float32)


# -------------------------------------------------------------------------
# Logistic Regression + hyper-parameter tuning on the VALIDATION split
# -------------------------------------------------------------------------
def tune_logreg(X_train, y_train, X_val, y_val):
    """
    Tune only on the validation set (never touch test), as required by the
    checklist. We use a small grid over C and the solver's regularisation.
    Embeddings are already L2-normalised, so no further scaling is needed.
    """
    grid = [
        {"C": c, "class_weight": None}
        for c in (0.5, 1.0, 2.0, 4.0, 8.0)
    ]

    best = None
    for params in grid:
        clf = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            solver="lbfgs",
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_f1   = f1_score(y_val, val_pred, average="macro")
        val_acc  = accuracy_score(y_val, val_pred)
        print(f"[tune] C={params['C']:<4}  val_acc={val_acc:.4f}  val_macroF1={val_f1:.4f}")
        if best is None or val_f1 > best["val_f1"]:
            best = {
                "params" : params,
                "val_f1" : val_f1,
                "val_acc": val_acc,
                "model"  : clf,
            }
    print(f"[tune] BEST  C={best['params']['C']}  "
          f"val_macroF1={best['val_f1']:.4f}  val_acc={best['val_acc']:.4f}")
    return best


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def score(y_true, y_pred, tag):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"[{tag}] accuracy={acc:.4f}  macroF1={f1m:.4f}")
    return {"accuracy": float(acc), "macro_f1": float(f1m)}


def full_report(y_true, y_pred, tag):
    print(f"\n[{tag}] classification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[LABEL_NAMES[i] for i in range(4)],
        digits=4,
    ))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"[{tag}] confusion matrix (rows=true, cols=pred, order=World,Sports,Business,Sci/Tech):")
    print(cm)
    return cm.tolist()


# -------------------------------------------------------------------------
# "Try it" — the checklist asks for a predict function that accepts a brand
#  new text, handles empty input, and does NOT retrain.
# -------------------------------------------------------------------------
def predict_text(text: str, embedder, clf):
    """Predict a single new text. Returns a dict so it is easy to print/save."""
    if text is None or not str(text).strip():
        return {
            "text" : text,
            "ok"   : False,
            "error": "Empty input: please enter a non-empty headline.",
        }
    cleaned = light_clean(text)
    emb     = embedder.encode(
        [cleaned],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    pred_id = int(clf.predict(emb)[0])
    probs   = clf.predict_proba(emb)[0]
    return {
        "text"      : text,
        "cleaned"   : cleaned,
        "ok"        : True,
        "pred_id"   : pred_id,
        "pred_label": LABEL_NAMES[pred_id],
        "probs"     : {LABEL_NAMES[i]: float(probs[i]) for i in range(4)},
    }


# Two illustrative draft examples OUTSIDE the AG News test set.
# The student must review/approve or rewrite and manually label personal originals.
# These examples do not substitute for individually confirmed course examples.
TRY_IT_EXAMPLES = [
    {
        "text"       : "Startup unveils AI chip that triples training speed",
        "true_label" : "Sci/Tech",
        "note"       : "Draft example; review and label personally before submission.",
    },
    {
        "text"       : "National team wins basketball championship after overtime thriller",
        "true_label" : "Sports",
        "note"       : "Draft example; review and label personally before submission.",
    },
]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--out_dir",   type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--model_dir", type=Path, default=Path(__file__).resolve().parent / "models")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Member 3 — Pretrained Sentence-Transformer + Logistic Regression")
    print("=" * 72)

    # 1. Load the data prepared by Member 1
    print("\n[1/6] Loading prepared data from", args.data_dir)
    train = load_split(args.data_dir, "train")
    val   = load_split(args.data_dir, "val")
    raw_test = pd.read_csv(args.data_dir / "test.csv")
    test  = load_split(args.data_dir, "test")
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"      short headlines  -> "
          f"train={int(train['is_short'].sum()):,}  "
          f"val={int(val['is_short'].sum()):,}  "
          f"test={int(test['is_short'].sum()):,}")

    # 2. Build the embedder and encode every split
    print("\n[2/6] Building sentence embedder")
    embedder = build_embedder()

    print("\n[3/6] Encoding splits")
    X_train = encode(embedder, train["text"].tolist(), args.batch_size)
    X_val   = encode(embedder, val["text"].tolist(),   args.batch_size)
    X_test  = encode(embedder, test["text"].tolist(),  args.batch_size)

    y_train = train["y"].to_numpy()
    y_val   = val["y"].to_numpy()
    y_test  = test["y"].to_numpy()

    # 3. Hyper-parameter tuning on validation
    print("\n[4/6] Tuning Logistic Regression on validation")
    best = tune_logreg(X_train, y_train, X_val, y_val)
    clf  = best["model"]

    # 4. Evaluate on validation, test, and the short-only subset of test
    print("\n[5/6] Evaluating best model")
    val_pred  = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    short_mask = test["is_short"].to_numpy().astype(bool)
    test_pred_short = test_pred[short_mask]
    y_test_short    = y_test[short_mask]

    metrics = {
        "model_id"        : EMBED_MODEL_ID,
        "classifier"      : "LogisticRegression(solver=lbfgs, max_iter=2000)",
        "best_hyperparams": best["params"],
        "val"             : score(y_val,         val_pred,        "val"),
        "test"            : score(y_test,        test_pred,       "test"),
        "test_short"      : score(y_test_short,  test_pred_short, "test-short"),
        "n_val"           : int(len(y_val)),
        "n_test"          : int(len(y_test)),
        "n_test_short"    : int(short_mask.sum()),
    }
    metrics["test"]["confusion_matrix"]       = full_report(y_test,       test_pred,       "test")
    metrics["test_short"]["confusion_matrix"] = full_report(y_test_short, test_pred_short, "test-short")

    # 5. Save predictions and model
    print("\n[6/6] Saving artefacts")
    val_out = val.copy()
    val_out["pred_id"]    = val_pred
    val_out["pred_label"] = val_out["pred_id"].map(LABEL_NAMES)
    val_out["true_label"] = val_out["y"].map(LABEL_NAMES)
    val_out[["text", "y", "true_label", "pred_id", "pred_label",
             "word_count", "is_short"]].to_csv(
        args.out_dir / "predictions_B_val.csv", index=False
    )

    test_out = test.copy()
    test_out["pred_id"]    = test_pred
    test_out["pred_label"] = test_out["pred_id"].map(LABEL_NAMES)
    test_out["true_label"] = test_out["y"].map(LABEL_NAMES)
    test_out[["text", "y", "true_label", "pred_id", "pred_label",
              "word_count", "is_short"]].to_csv(
        args.out_dir / "predictions_B.csv", index=False
    )

    with open(args.out_dir / "metrics_B.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Export canonical results in the test.csv row order and 1..4 label scheme.
    assert len(raw_test) == len(test)
    assert raw_test['text'].map(light_clean).equals(test['text'])
    assert (raw_test['label'].to_numpy() - 1 == test['y'].to_numpy()).all()
    pd.DataFrame({'text': raw_test['text'], 'prediction': test_pred + 1}).to_csv(
        args.out_dir / 'minilm_predictions.csv', index=False
    )
    # An inference-friendly weight export avoids cross-version sklearn pickle loading.
    import numpy as _np
    _np.savez_compressed(args.model_dir / 'minilm_logreg_coefficients.npz',
                         coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)
    joblib.dump(clf, args.model_dir / "pretrained_lr.joblib")
    (args.model_dir / "embedder_name.txt").write_text(EMBED_MODEL_ID + "\n")

    # 6. The two Try-It examples (checklist item)
    try_rows = []
    print("\n[try-it] running the two manual examples:")
    for ex in TRY_IT_EXAMPLES:
        r = predict_text(ex["text"], embedder, clf)
        print(f"  '{ex['text']}'  ->  true={ex['true_label']}  "
              f"pred={r['pred_label']}  "
              f"conf={r['probs'][r['pred_label']]:.3f}")
        try_rows.append({
            "text"      : ex["text"],
            "true_label": ex["true_label"],
            "pred_label": r["pred_label"],
            "confidence": r["probs"][r["pred_label"]],
            "note"      : ex["note"],
        })

    # Empty-input guard, also required by the checklist
    r_empty = predict_text("", embedder, clf)
    print(f"  (empty input) -> ok={r_empty['ok']}  msg={r_empty.get('error')}")
    try_rows.append({
        "text"      : "",
        "true_label": "-",
        "pred_label": "-",
        "confidence": 0.0,
        "note"      : r_empty.get("error", ""),
    })

    pd.DataFrame(try_rows).to_csv(
        args.out_dir / "try_it_examples_B.csv", index=False
    )

    print("\nDone. Artefacts written under:",
          args.out_dir, "and", args.model_dir)


if __name__ == "__main__":
    main()
