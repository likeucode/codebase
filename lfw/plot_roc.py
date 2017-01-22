# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:47:32 2017

@author: Ke
"""

from scipy.io import loadmat
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

result=loadmat('C:/Users/User/Desktop/result_pca_model2.mat')

dis=result['distance']
label=result['label'][0,:]
fpr, tpr, thresholds = roc_curve(label, dis.T)

fnr = 1-tpr
resd=np.abs(fnr-fpr)
eer_idx=np.argmin(resd)
eer = fpr[eer_idx]
print ("EER :",eer)



lw=2
roc_auc = auc(fpr, tpr)
print("AUC :", roc_auc)

plt.plot(fpr, tpr, lw=lw, color='blue',label='area = %0.2f' % roc_auc)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Joint Bayes ROC in LFW')
plt.legend(loc="lower right")
plt.grid(True)
plt.xticks(np.arange(0,1.05,0.1))
plt.yticks(np.arange(0,1.05,0.1))
plt.show()