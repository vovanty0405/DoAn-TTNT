"""Reference records are used ONLY to verify meaning, never for experiments."""
import sys, hashlib, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from heart_pipeline import ROOT,OUT,FEATURES,write_json

def main():
 OUT.mkdir(exist_ok=True);evidence=ROOT/'.build/provenance';evidence.mkdir(parents=True,exist_ok=True)
 url='https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'
 for name in ['processed.cleveland.data','heart-disease.names']:
  destination=evidence/name
  if not destination.exists(): urllib.request.urlretrieve(url+name,destination)
 raw=pd.read_csv(ROOT/'Data/heart.csv').drop_duplicates()
 ref=pd.read_csv(evidence/'processed.cleveland.data',header=None,names=FEATURES+['target'],na_values='?')
 keys=['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak']
 assert not ref.duplicated(keys).any() and not raw.duplicated(keys).any()
 matched=raw.merge(ref,on=keys,suffixes=('_heart','_uci'),validate='one_to_one');assert len(matched)==len(raw)
 agreement=int(((1-matched.target_heart)==matched.target_uci.gt(0).astype(int)).sum());assert agreement==len(raw)
 maps={'cp':{0:4,1:2,2:3,3:1},'restecg':{0:2,1:0,2:1},'slope':{0:3,1:2,2:1},'thal':{1:6,2:3,3:7}}
 for col,mapping in maps.items():
  available=matched[col+'_uci'].notna() & matched[col+'_heart'].isin(mapping)
  assert (matched.loc[available,col+'_heart'].map(mapping)==matched.loc[available,col+'_uci']).all()
 write_json(OUT/'label_verification.json',{'heart_sha256':hashlib.sha256((ROOT/'Data/heart.csv').read_bytes()).hexdigest(),
  'reference_sha256':hashlib.sha256((evidence/'processed.cleveland.data').read_bytes()).hexdigest(),
  'reference_url':url+'processed.cleveland.data','metadata_url':url+'heart-disease.names','matched_rows':len(matched),'label_agreement':agreement,
  'match_keys':keys,'category_maps_heart_to_uci':maps,'purpose':'Verify target/category meaning ONLY. Never fit, validate or evaluate with reference records.',
  'special_missing_policy':'ca=4 and thal=0 become NaN. ca=4 is not universally identical to missing in the original reference.'})
 print(f'Verified {agreement}/{len(raw)} labels and categorical mappings.')

if __name__=='__main__':main()
