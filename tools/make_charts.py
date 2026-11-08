"""Reproducible empirical charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from heart_pipeline import OUT,LABELS

def make_charts(data,audit,details,selected):
 charts=OUT/'charts';charts.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
 colors=['#315B65','#CC6845']
 def save(name):
  plt.tight_layout();plt.savefig(charts/(name+'.png'),dpi=200,bbox_inches='tight');plt.close()
 counts=data.target.value_counts().sort_index()
 fig,ax=plt.subplots(figsize=(6,3.6));bars=ax.bar([LABELS[i] for i in counts.index],counts.values,color=colors)
 ax.bar_label(bars);ax.set_ylabel('Số bản ghi duy nhất');ax.set_title('Phân bố nhãn sau chuẩn hóa');ax.set_ylim(0,counts.max()*1.2);save('labels')
 rates=data.groupby('cp').target.agg(['mean','count']);rates.to_csv(OUT/'cp_rates.csv')
 fig,ax=plt.subplots(figsize=(6,3.6));bars=ax.bar(['Không triệu\nchứng','Không điển\nhình','Không do\nthắt ngực','Điển hình'],rates['mean']*100,color=colors[1])
 ax.bar_label(bars,labels=[f'{v:.1f}%\nn={n}' for v,n in zip(rates['mean']*100,rates['count'])],fontsize=10)
 ax.set_ylim(0,100);ax.set_ylabel('Tỷ lệ có bệnh (%)');ax.set_title('Nhãn có bệnh theo loại đau ngực');save('cp_rates')
 for col,title,unit in [('age','Độ tuổi','năm'),('chol','Cholesterol','mg/dL'),('thalach','Nhịp tim tối đa','nhịp/phút')]:
  fig,ax=plt.subplots(figsize=(6,3.6))
  for label,color in zip([0,1],colors): ax.hist(data.loc[data.target.eq(label),col],bins=14,alpha=.6,label=LABELS[label],color=color)
  ax.set_title(title);ax.set_xlabel(unit);ax.set_ylabel('Số bản ghi');ax.legend();save('distribution_'+col)
  fig,ax=plt.subplots(figsize=(5,3.6));ax.boxplot([data.loc[data.target.eq(i),col] for i in [0,1]],tick_labels=[LABELS[0],LABELS[1]])
  ax.set_ylabel(unit);ax.set_title(title+' theo nhãn');save('box_'+col)
 data.groupby('target')[['age','chol','thalach','oldpeak']].agg(['mean','median','std']).to_csv(OUT/'group_statistics.csv')
 for name,result in details.items():
  fig,ax=plt.subplots(figsize=(5,3.8));ConfusionMatrixDisplay(np.array(result['test']['confusion_matrix']),display_labels=['Không bệnh','Có bệnh']).plot(ax=ax,cmap='Blues',colorbar=False)
  ax.set_title(name+' – Test');ax.set_xlabel('Dự đoán');ax.set_ylabel('Thực tế');save('cm_'+name.replace(' ','_'))
 metrics=pd.read_csv(OUT/'metrics.csv');fig,ax=plt.subplots(figsize=(7,3.6));x=np.arange(len(metrics))
 for j,metric in enumerate(['accuracy','recall','f1']):ax.bar(x+(j-1)*.24,metrics[metric],.24,label=metric)
 ax.set_xticks(x,metrics.model);ax.set_ylim(0,1.1);ax.set_ylabel('Điểm trên Test');ax.legend(ncol=3);ax.set_title('Hiệu năng ba mô hình');save('performance')
 imp=pd.read_csv(OUT/'permutation_importance.csv').head(7).iloc[::-1]
 fig,ax=plt.subplots(figsize=(6,3.8));ax.barh(imp.feature,imp['mean'],xerr=imp['std'],color=colors[0]);ax.axvline(0,color='gray',lw=.8)
 ax.set_xlabel('Mức giảm F1 (trung bình ± SD, 30 lần)');ax.set_title('Permutation importance trên Test');save('importance')
