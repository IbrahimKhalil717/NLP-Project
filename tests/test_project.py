"""Offline verification tests for the frozen approved experiment.
Run from project root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,f1_score
from prepare_minilm_predictions import convert
from predict_news import predict_tfidf, predict_minilm
from evaluation import evaluate

ROOT=Path(__file__).resolve().parent.parent

class ProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=pd.read_csv(ROOT/'data/train.csv')
        cls.val=pd.read_csv(ROOT/'data/val.csv')
        cls.test=pd.read_csv(ROOT/'data/test.csv')
        cls.a=pd.read_csv(ROOT/'results/tfidf_predictions.csv')
        cls.b=pd.read_csv(ROOT/'results/minilm_predictions.csv')

    def test_splits(self):
        self.assertEqual([len(self.train),len(self.val),len(self.test)],[101043,11227,7464])
        K=lambda df:set(df.text.str.replace(r'\s+',' ',regex=True).str.strip().str.casefold())
        t,v,s=K(self.train),K(self.val),K(self.test)
        self.assertFalse(t&v);self.assertFalse(t&s);self.assertFalse(v&s)
        self.assertEqual(int(self.test.is_short.sum()),3554)
        self.assertTrue(self.test.is_short.eq(self.test.word_count.le(6)).all())

    def test_frozen_predictions_row_alignment(self):
        self.assertEqual(len(self.a),len(self.test))
        self.assertEqual(len(self.b),len(self.test))
        self.assertTrue(self.a.text.equals(self.test.text))
        self.assertTrue(self.b.text.equals(self.test.text))
        self.assertTrue(self.a.prediction.isin([1,2,3,4]).all())
        self.assertTrue(self.b.prediction.isin([1,2,3,4]).all())

    def test_metrics_recalculated(self):
        self.assertAlmostEqual(accuracy_score(self.test.label,self.a.prediction),0.8829046087888531,places=10)
        self.assertAlmostEqual(accuracy_score(self.test.label,self.b.prediction),0.8381564844587353,places=10)
        self.assertAlmostEqual(f1_score(self.test.label,self.a.prediction,average='macro'),0.8822607837461339,places=10)
        self.assertAlmostEqual(f1_score(self.test.label,self.b.prediction,average='macro'),0.8375106327923971,places=10)
        m=self.test.is_short
        self.assertAlmostEqual(f1_score(self.test.loc[m,'label'],self.a.loc[m,'prediction'],average='macro'),0.8655604500754228,places=10)
        self.assertAlmostEqual(f1_score(self.test.loc[m,'label'],self.b.loc[m,'prediction'],average='macro'),0.8094538548776562,places=10)

    def test_tfidf_prediction_reproduction(self):
        trusted=joblib.load(ROOT/'outputs/tfidf/tfidf_logreg.joblib')
        check_indices=[0,25,101,400,799,1200,4491,7463]
        predicted=trusted.predict(self.test.text.iloc[check_indices])
        np.testing.assert_array_equal(predicted,self.a.prediction.iloc[check_indices].to_numpy())
        self.assertEqual(predict_tfidf('  ')['ok'],False)
        self.assertEqual(predict_tfidf('Underdog team clinches regional football title')['id'],2)
        self.assertEqual(predict_tfidf('Scientists unveil faster battery charging method')['id'],4)

    def test_minilm_conversion_verified(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'minilm_predictions.csv'
            d=convert(ROOT/'data/test.csv', ROOT/'results/predictions_B.csv',target)
            self.assertEqual(d['rows'],7464)
            self.assertEqual(d['text_changed_by_minilm_preprocessing'],151)
            pd.testing.assert_frame_equal(pd.read_csv(target),self.b)

    def test_minilm_weight_export(self):
        with np.load(ROOT/'models/minilm_logreg_coefficients.npz') as data:
            self.assertEqual(data['coef'].shape,(4,384))
            self.assertEqual(list(data['classes']),[0,1,2,3])
        self.assertEqual(predict_minilm('  ')['ok'],False)

    def test_team_evaluator_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            m,e=evaluate(ROOT/'data/test.csv',ROOT/'results/tfidf_predictions.csv',
                        ROOT/'results/minilm_predictions.csv',Path(folder))
            self.assertEqual(len(m),4)
            self.assertEqual(len(e),1498)
            self.assertEqual(int(e.is_short.sum()),self._short_error_count(e))
            self.assertTrue((Path(folder)/'metrics.csv').exists())
    @staticmethod
    def _short_error_count(df):return len(df[df.is_short])

if __name__=='__main__': unittest.main()
