"""Classify unfamiliar English news headlines with the *saved* models.

Usage:
    python predict_news.py --model tfidf --text "Team wins football final"
    python predict_news.py --model minilm --text "New battery technology announced"

MiniLM downloads sentence-transformers/all-MiniLM-L6-v2 on first use.
No training occurs in this file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS={1:'World',2:'Sports',3:'Business',4:'Sci/Tech'}


def predict_tfidf(text:str)->dict:
    if not isinstance(text,str) or not text.strip():
        return {'ok':False,'message':'Please enter a non-empty English news headline.'}
    model=joblib.load(ROOT/'outputs/tfidf/tfidf_logreg.joblib')  # trusted project artifact
    label=int(model.predict([text.strip()])[0])
    return {'ok':True,'model':'TF-IDF + Logistic Regression','id':label,'category':LABELS[label]}


def predict_minilm(text:str,embedder=None)->dict:
    if not isinstance(text,str) or not text.strip():
        return {'ok':False,'message':'Please enter a non-empty English news headline.'}
    from pretrained_model import light_clean
    clean=light_clean(text)
    if not clean:
        return {'ok':False,'message':'Please enter a non-empty English news headline.'}
    if embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install requirements.txt to use MiniLM live predictions') from exc
        model_id=(ROOT/'models/embedder_name.txt').read_text(encoding='utf-8').strip()
        embedder=SentenceTransformer(model_id)
    x=embedder.encode([clean],convert_to_numpy=True,normalize_embeddings=True)
    weight=np.load(ROOT/'models/minilm_logreg_coefficients.npz')
    logits=np.asarray(x)@weight['coef'].T+weight['intercept']
    # 4-class multinomial LR: argmax of logits, mapped back to 1..4.
    label=int(weight['classes'][np.argmax(logits,axis=1)[0]])+1
    return {'ok':True,'model':'MiniLM + Logistic Regression','id':label,'category':LABELS[label]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tfidf','minilm'],default='tfidf')
    parser.add_argument('--text',required=True)
    a=parser.parse_args()
    result=predict_tfidf(a.text) if a.model=='tfidf' else predict_minilm(a.text)
    print(result)

if __name__=='__main__':
    main()
