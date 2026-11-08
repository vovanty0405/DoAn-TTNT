"""Integration checks against exported artifacts and actual Streamlit widgets."""
import sys,json,hashlib,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
from streamlit.testing.v1 import AppTest
from heart_pipeline import ROOT,OUT,FEATURES,load_heart,normalize_input,disease_score,evaluate

class ProjectTest(unittest.TestCase):
 def test_data(self):
  raw,data,audit=load_heart();self.assertEqual((len(raw),len(data)),(1025,302));self.assertEqual(audit['duplicates'],723)
  self.assertEqual(audit['conflicting_groups'],0)
  train=pd.read_csv(OUT/'train.csv');test=pd.read_csv(OUT/'test.csv')
  self.assertEqual((len(train),len(test)),(211,91))
  self.assertFalse(set(train.source_row)&set(test.source_row))
  self.assertFalse(set(map(tuple,train[FEATURES].fillna(-999).to_numpy()))&set(map(tuple,test[FEATURES].fillna(-999).to_numpy())))
  self.assertEqual(hashlib.sha256((ROOT/'Data/heart.csv').read_bytes()).digest(),hashlib.sha256((ROOT/'backups/original_20261105/heart.csv').read_bytes()).digest())

 def test_models(self):
  test=pd.read_csv(OUT/'test.csv');train=pd.read_csv(OUT/'train.csv');exp=json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
  for name,detail in exp['models'].items():
   model=joblib.load(OUT/'models'/(name.replace(' ','_')+'.joblib'))
   result=evaluate(model,test[FEATURES],test.target)
   for key in ['accuracy','precision','recall','f1','roc_auc']:self.assertAlmostEqual(result[key],detail['test'][key])
   self.assertEqual(list(model.classes_),[0,1]);self.assertEqual(model.n_features_in_,13)
   np.testing.assert_allclose(model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_,train[['age','trestbps','chol','thalach','oldpeak']].median().values)
   unknown=test[FEATURES].iloc[[0]].copy();unknown['ca']=4;unknown['thal']=0
   self.assertTrue(np.isfinite(disease_score(model,unknown)).all())
   np.testing.assert_allclose(model.predict_proba(unknown),model.predict_proba(normalize_input(unknown)))
   shuffled=test[list(reversed(FEATURES))]
   np.testing.assert_allclose(disease_score(model,shuffled),disease_score(model,test[FEATURES]))
  cv=pd.read_csv(OUT/'cross_validation.csv');self.assertEqual(len(cv),6)
  best=cv[cv.variant.eq('tuned')].sort_values(['f1_mean','model'],ascending=[False,True]).iloc[0].model
  self.assertEqual(exp['selected_model'],best)

 def test_app(self):
  app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
  self.assertFalse(app.exception)
  self.assertEqual(len(app.number_input)+len(app.selectbox)-2,13)
  for model in ['Decision Tree','Random Forest','Gaussian NB']:
   app.selectbox[0].select(model).run();app.button[0].click().run();self.assertFalse(app.exception)
   self.assertIn('theo dữ liệu nghiên cứu',app.success[0].value)
  app.selectbox[0].select('Random Forest').run()
  app.selectbox[1].select('Không có bệnh theo nhãn thật').run();app.button[0].click().run()
  self.assertFalse(app.exception);self.assertIn('không có bệnh',app.success[0].value)
  # Change numeric and special missing categories, preserving the real input flow.
  app.number_input[0].set_value(50.).run();app.button[0].click().run();self.assertFalse(app.exception)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectTest))
 (OUT/'test_report.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()},indent=2),encoding='utf-8')
 sys.exit(0 if result.wasSuccessful() else 1)
