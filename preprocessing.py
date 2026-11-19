"""Member 1: AG News headline-only preprocessing.
Run: python preprocessing.py
Requires: datasets, pandas, scikit-learn
"""
from pathlib import Path
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
LABELS = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
SEED = 42


def clean_title(value):
    """Conservative cleaning: retain words, numbers, punctuation and case."""
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def normalize_key(title):
    """Used only for duplicate detection, not as model input."""
    return clean_title(title).casefold()


def prepare_frame(frame):
    result = frame[['title', 'label']].copy()
    result['text'] = result['title'].map(clean_title)
    result = result[result['text'].ne('')].copy()
    result['label'] = result['label'].astype(int)
    if not result['label'].isin(LABELS).all():
        raise ValueError('Unexpected dataset label. Expected integers 1, 2, 3, 4.')
    result['key'] = result['text'].map(normalize_key)
    result['word_count'] = result['text'].str.split().str.len()
    result['is_short'] = result['word_count'].le(6)
    return result[['text', 'label', 'key', 'word_count', 'is_short']]


def process_splits(raw_train, raw_test):
    train = prepare_frame(raw_train)
    test = prepare_frame(raw_test)
    stats = {'raw_train': len(raw_train), 'raw_test': len(raw_test)}

    # Ambiguous headlines (same title but multiple labels) are excluded.
    all_rows = pd.concat([train, test], ignore_index=True)
    inconsistent = set(all_rows.groupby('key')['label'].nunique().loc[lambda x: x > 1].index)
    stats['ambiguous_headline_keys_removed'] = len(inconsistent)
    train = train[~train['key'].isin(inconsistent)].copy()
    test = test[~test['key'].isin(inconsistent)].copy()

    # Keep one example per headline; hold the original test split out.
    train = train.drop_duplicates('key').copy()
    test = test.drop_duplicates('key').copy()
    test_keys = set(test['key'])
    overlap = train['key'].isin(test_keys)
    stats['train_rows_overlapping_test_removed'] = int(overlap.sum())
    train = train[~overlap].copy()

    # Validation is a stratified 10% hold-out from original training data.
    training, validation = train_test_split(
        train, test_size=0.1, random_state=SEED, stratify=train['label']
    )
    training, validation, test = [f.reset_index(drop=True) for f in (training, validation, test)]
    assert not (set(training['key']) & set(validation['key']))
    assert not (set(training['key']) & set(test['key']))
    assert not (set(validation['key']) & set(test['key']))

    for name, frame in [('train', training), ('val', validation), ('test', test)]:
        stats[name] = {
            'count': len(frame),
            'short_count': int(frame['is_short'].sum()),
            'class_counts': {LABELS[k]: int(v) for k, v in frame['label'].value_counts().sort_index().items()}
        }
    return training, validation, test, stats


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset('sh0416/ag_news')
    train, val, test, stats = process_splits(ds['train'].to_pandas(), ds['test'].to_pandas())
    for name, frame in [('train', train), ('val', val), ('test', test)]:
        frame.drop(columns='key').to_csv(DATA_DIR / f'{name}.csv', index=False)
    (DATA_DIR / 'preprocessing_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
    print('Created data/train.csv, data/val.csv, data/test.csv')


if __name__ == '__main__':
    main()
