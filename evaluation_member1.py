"""AG News Group 5 — Member 1 evaluation (headline-only, 1..4 labels).

Expected teammate prediction files:
    results/tfidf_predictions.csv
    results/minilm_predictions.csv
Each file must contain a 'prediction' column and preferably a 'text' column
copied from data/test.csv in exactly the same row order.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABEL_NAMES = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
CLASSES = list(LABEL_NAMES)


def read_predictions(file_path, test):
    """Load predictions, check label range and (if supplied) text-row alignment."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'Missing {file_path}. Ask your teammate to export test predictions.')
    df = pd.read_csv(file_path)
    if 'prediction' not in df:
        raise ValueError(f'{file_path}: expected a column named prediction.')
    if len(df) != len(test):
        raise ValueError(f'{file_path}: {len(df)} rows, expected {len(test)} test rows.')
    if df['prediction'].isna().any():
        raise ValueError(f'{file_path}: predictions contain missing values.')
    numeric = pd.to_numeric(df['prediction'], errors='coerce')
    if numeric.isna().any() or not numeric.isin(CLASSES).all():
        raise ValueError(f'{file_path}: all predictions must be integers 1, 2, 3, or 4.')
    if 'text' in df and not df['text'].fillna('').astype(str).equals(test['text'].fillna('').astype(str)):
        raise ValueError(f'{file_path}: text rows do not match test.csv. Do not shuffle test predictions.')
    if 'text' not in df:
        print(f'WARNING: {file_path.name} has no text column; row alignment cannot be verified.')
    return numeric.astype(int)


def evaluate(test_path='data/test.csv', tfidf_path='results/tfidf_predictions.csv',
             minilm_path='results/minilm_predictions.csv', output_dir='results'):
    test = pd.read_csv(test_path)
    needed = {'text', 'label', 'word_count', 'is_short'}
    if not needed.issubset(test):
        raise ValueError(f'Test CSV missing columns: {sorted(needed - set(test.columns))}')
    if test[['text', 'label', 'word_count', 'is_short']].isna().any().any():
        raise ValueError('Test CSV contains missing values.')
    if not pd.to_numeric(test['label'], errors='coerce').isin(CLASSES).all():
        raise ValueError('Unexpected ground-truth labels: expected 1..4.')
    test['label'] = test['label'].astype(int)
    expected_short = test['word_count'].le(6)
    parsed_short = test['is_short'].astype(str).str.lower().map({'true': True, 'false': False})
    if parsed_short.isna().any() or not parsed_short.equals(expected_short):
        raise ValueError('is_short must indicate word_count <= 6.')

    preds = {
        'TF-IDF + Logistic Regression': read_predictions(tfidf_path, test),
        'MiniLM + Logistic Regression': read_predictions(minilm_path, test),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for name, y_pred in preds.items():
        for subset_name, mask in [('All headlines', pd.Series(True, index=test.index)),
                                  ('Short headlines (<=6 words)', expected_short)]:
            yt = test.loc[mask, 'label']
            yp = y_pred.loc[mask]
            if not len(yt):
                raise ValueError(f'No examples in {subset_name}')
            metrics_rows.append({
                'model': name, 'subset': subset_name, 'n': len(yt),
                'accuracy': accuracy_score(yt, yp),
                'macro_f1': f1_score(yt, yp, labels=CLASSES, average='macro', zero_division=0),
            })
        safe_name = 'tfidf' if name.startswith('TF-IDF') else 'minilm'
        report = classification_report(test['label'], y_pred, labels=CLASSES,
                                       target_names=[LABEL_NAMES[i] for i in CLASSES], zero_division=0)
        (output_dir / f'{safe_name}_classification_report.txt').write_text(report, encoding='utf-8')
        cm = pd.DataFrame(confusion_matrix(test['label'], y_pred, labels=CLASSES),
                          index=[f'actual_{LABEL_NAMES[i]}' for i in CLASSES],
                          columns=[f'predicted_{LABEL_NAMES[i]}' for i in CLASSES])
        cm.to_csv(output_dir / f'{safe_name}_confusion_matrix.csv')

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / 'metrics.csv', index=False)
    analysis = test.copy()
    analysis['tfidf_prediction'] = preds['TF-IDF + Logistic Regression'].to_numpy()
    analysis['minilm_prediction'] = preds['MiniLM + Logistic Regression'].to_numpy()
    analysis['tfidf_correct'] = analysis['tfidf_prediction'].eq(analysis['label'])
    analysis['minilm_correct'] = analysis['minilm_prediction'].eq(analysis['label'])
    errors = analysis.loc[~analysis['tfidf_correct'] | ~analysis['minilm_correct']].copy()
    for column in ['label', 'tfidf_prediction', 'minilm_prediction']:
        errors[column + '_name'] = errors[column].map(LABEL_NAMES)
    errors.to_csv(output_dir / 'errors_to_analyze.csv', index=False)
    print('\nEvaluation completed. Results:')
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nHeadlines with at least one incorrect prediction: {len(errors)}')
    print(f'Files saved to: {output_dir.resolve()}')
    return metrics, errors


if __name__ == '__main__':
    evaluate()
