"""Fair, reproducible evaluation of the two frozen classifiers on one test split.

Usage: python evaluation.py

Predictions must match *exactly* the ordered rows of data/test.csv, use labels
1..4 and be frozen before test evaluation. No training or parameter tuning is
performed here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
LABEL_NAMES = {1:'World',2:'Sports',3:'Business',4:'Sci/Tech'}
CLASSES=[1,2,3,4]


def _read_prediction(path: Path, test:pd.DataFrame)->pd.Series:
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; export the frozen predictions first')
    frame=pd.read_csv(path)
    if not {'text','prediction'}.issubset(frame.columns):
        raise ValueError(f'{path.name} must contain text and prediction columns')
    if len(frame)!=len(test) or not frame['text'].equals(test['text']):
        raise ValueError(f'{path.name} headline rows do not match data/test.csv in original order')
    p=pd.to_numeric(frame['prediction'],errors='raise')
    if p.isna().any() or not p.isin(CLASSES).all() or not (p.astype(int)==p).all():
        raise ValueError(f'{path.name} predictions must be integers from 1 to 4')
    return p.astype(int).reset_index(drop=True)


def evaluate(test_path=ROOT/'data/test.csv', tfidf_path=ROOT/'results/tfidf_predictions.csv',
             minilm_path=ROOT/'results/minilm_predictions.csv', output_dir=ROOT/'results'):
    test=pd.read_csv(test_path).reset_index(drop=True)
    columns={'text','label','word_count','is_short'}
    if not columns.issubset(test):
        raise ValueError(f'Test data missing: {columns-set(test)}')
    if test[list(columns)].isna().any().any():
        raise ValueError('Missing value in test data')
    test['label']=test['label'].astype(int)
    if not test['label'].isin(CLASSES).all():
        raise ValueError('Test labels must be 1..4')
    expected_short=test.text.str.split().str.len().le(6)
    if not test['word_count'].equals(test.text.str.split().str.len()):
        raise ValueError('Test word_count column does not match original text')
    flags=test['is_short'].astype(str).str.casefold().map({'true':True,'false':False})
    if flags.isna().any() or not flags.equals(expected_short):
        raise ValueError('Test is_short column does not match six-word definition')
    preds={
        'TF-IDF + Logistic Regression':_read_prediction(Path(tfidf_path),test),
        'MiniLM + Logistic Regression':_read_prediction(Path(minilm_path),test),
    }
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    masks={'all':pd.Series(True,index=test.index),'short_6_or_less':expected_short}
    rows=[]
    for name,pred in preds.items():
        prefix='tfidf' if name.startswith('TF-IDF') else 'minilm'
        for subset,mask in masks.items():
            y=test.loc[mask,'label'];p=pred.loc[mask]
            rows.append({'model':name,'subset':subset,'n':len(y),
                         'accuracy':float(accuracy_score(y,p)),
                         'macro_f1':float(f1_score(y,p,labels=CLASSES,average='macro',zero_division=0))})
        cm=pd.DataFrame(confusion_matrix(test.label,pred,labels=CLASSES),
                        index=[f'actual_{LABEL_NAMES[i]}' for i in CLASSES],
                        columns=[f'predicted_{LABEL_NAMES[i]}' for i in CLASSES])
        cm.to_csv(output/f'{prefix}_confusion_matrix.csv')
        (output/f'{prefix}_classification_report.txt').write_text(
            classification_report(test.label,pred,labels=CLASSES,
                target_names=[LABEL_NAMES[i] for i in CLASSES],zero_division=0),encoding='utf-8')
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'metrics.csv',index=False)
    analysis=test.copy()
    analysis['tfidf_prediction']=preds['TF-IDF + Logistic Regression']
    analysis['minilm_prediction']=preds['MiniLM + Logistic Regression']
    analysis['tfidf_correct']=analysis['tfidf_prediction'].eq(analysis.label)
    analysis['minilm_correct']=analysis['minilm_prediction'].eq(analysis.label)
    analysis['error_pattern']=np.select(
        [~analysis.tfidf_correct & ~analysis.minilm_correct,
         ~analysis.tfidf_correct & analysis.minilm_correct,
         analysis.tfidf_correct & ~analysis.minilm_correct],
        ['both_wrong','tfidf_only_wrong','minilm_only_wrong'],default='both_correct')
    for col in ('label','tfidf_prediction','minilm_prediction'):
        analysis[col+'_name']=analysis[col].map(LABEL_NAMES)
    errors=analysis.loc[analysis.error_pattern!='both_correct'].copy()
    errors.to_csv(output/'errors_to_analyze.csv',index=False)
    errors.loc[errors.is_short].to_csv(output/'short_headline_errors.csv',index=False)
    base=frame.pivot(index='subset',columns='model',values=['accuracy','macro_f1'])
    summary={'research_question':'Which method performs better on headlines containing six words or fewer?',
             'splits':{'test':len(test),'test_short':int(expected_short.sum())},
             'comparison':[dict(r) for r in frame.to_dict(orient='records')],
             'error_patterns':analysis.error_pattern.value_counts().to_dict(),
             'short_error_patterns':analysis.loc[expected_short,'error_pattern'].value_counts().to_dict()}
    (output/'comparison_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(frame.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nError categories:',summary['error_patterns'])
    print('Results saved:',output)
    return frame,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',type=Path,default=ROOT/'data/test.csv')
    parser.add_argument('--tfidf',type=Path,default=ROOT/'results/tfidf_predictions.csv')
    parser.add_argument('--minilm',type=Path,default=ROOT/'results/minilm_predictions.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    evaluate(args.test,args.tfidf,args.minilm,args.output)

if __name__=='__main__':
    main()
