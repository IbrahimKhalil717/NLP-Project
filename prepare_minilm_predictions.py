"""Create the team's canonical MiniLM prediction file from the frozen results.

Usage: python prepare_minilm_predictions.py

Does NOT retrain, tune or select a model on test. The source prediction file
was produced before this adapter. Its 0..3 labels are shifted to 1..4 and the
original test text is restored without changing predictions or row order.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from pretrained_model import light_clean

ROOT = Path(__file__).resolve().parent


def convert(test_path: Path, source_path: Path, output_path: Path) -> dict:
    test = pd.read_csv(test_path)
    old = pd.read_csv(source_path)
    if not {'text', 'label'}.issubset(test.columns):
        raise ValueError('Test file needs text and label columns')
    if not {'text', 'y', 'pred_id'}.issubset(old.columns):
        raise ValueError('MiniLM original results need text, y, pred_id')
    if len(test) != len(old):
        raise ValueError(f'Row-count mismatch: {len(test)} vs {len(old)}')
    if test.text.isna().any() or old.text.isna().any():
        raise ValueError('Missing text in test/prediction file')
    true_labels = pd.to_numeric(test['label'], errors='raise').astype('int64')
    y = pd.to_numeric(old['y'], errors='raise').astype('int64')
    pred = pd.to_numeric(old['pred_id'], errors='raise').astype('int64')
    if not true_labels.isin([1,2,3,4]).all() or not y.isin([0,1,2,3]).all() or not pred.isin([0,1,2,3]).all():
        raise ValueError('Unexpected class IDs: expected test 1..4, source 0..3')
    if not (true_labels.to_numpy() - 1 == y.to_numpy()).all():
        raise ValueError('Target label mismatch; source file may be shuffled')
    # MiniLM originally ran light_clean() on input text. Verify every row,
    # not only the target label, to detect row shuffling/corruption.
    expected = test['text'].map(light_clean)
    if not expected.equals(old['text']):
        mismatches = (expected != old['text'])
        i = mismatches[mismatches].index[0]
        raise ValueError(f'Headline order or preprocessing mismatch at row {i}')
    aligned = pd.DataFrame({'text':test['text'], 'prediction':pred+1})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    n_cleaned = int((test['text'] != old['text']).sum())
    return {'rows':len(aligned), 'text_changed_by_minilm_preprocessing':n_cleaned, 'saved':str(output_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',type=Path,default=ROOT/'data/test.csv')
    parser.add_argument('--source',type=Path,default=ROOT/'results/predictions_B.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'results/minilm_predictions.csv')
    a=parser.parse_args()
    print(convert(a.test,a.source,a.output))

if __name__=='__main__':
    main()
