##### Script for Muon Energy Reconstruction in the water tank
#import Store
import sys
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
import pickle
import seaborn as sns

#import ROOT
#ROOT.gROOT.SetBatch(True)
#from ROOT import TFile, TNtuple
#from root_numpy import root2array, tree2array, fill_hist

#-------- File with events for reconstruction:
#--- evts for training:
#infile = "../../LocalFolder/vars_Ereco.csv"
#infile = "/Users/edrakopo/work/ANNIEReco_PythonScripts/vars_Ereco_05202019.csv"
#--- evts for prediction:
#infile2 = "../../LocalFolder/vars_Ereco.csv"
#infile2 = "../TrackLengthReconstruction/vars_Ereco_pred_05202019.csv"
infile2 = "vars_Ereco_pred.csv"
#infile2 = "../TrackLengthReconstruction/vars_Ereco_pred_06082019CC0pi.csv"
#----------------

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

E_threshold = 2.
E_low=0
E_high=2000
div=100
bins = int((E_high-E_low)/div)
print('bins: ', bins)

print( "--- opening file with input variables!") 

#--- events for predicting ---
filein2 = open(str(infile2))
print(filein2)
df00b = pd.read_csv(filein2)
df0b=df00b[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','TrueTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel_pred=df0b.loc[df0b['neutrinoE'] < E_threshold]
dfsel_pred=df0b
#print to check:
print("check predicting sample: ",dfsel_pred.shape," ",dfsel_pred.head())
#   print(dfsel_pred.iloc[5:10,0:5])
#check fr NaN values:
assert(dfsel_pred.isnull().any().any()==False)

#--- normalisation-sample for prediction:
#dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd']/200., dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR']/152.4, dfsel_pred['recoDWallZ']/198., dfsel_pred['totalLAPPDs']/1000., dfsel_pred['totalPMTs']/1000., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T
dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd'], dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR'], dfsel_pred['recoDWallZ'], dfsel_pred['totalLAPPDs']/200., dfsel_pred['totalPMTs']/200., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T

#prepare events for predicting 
evts_to_predict_n= np.array(dfsel_pred_n[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs', 'vtxX','vtxY','vtxZ']])
test_data_trueKE_hi_E = np.array(dfsel_pred[['trueKE']])

n_estimators=1000

# load the model from disk
filename = 'finalized_BDTmodel_forMuonEnergy.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

#predicting...
print("events for energy reco: ", len(evts_to_predict_n)) 
BDTGoutput_E = loaded_model.predict(evts_to_predict_n)

Y=[0 for j in range (0,len(test_data_trueKE_hi_E))]
for i in range(len(test_data_trueKE_hi_E)):
    Y[i] = 100.*(test_data_trueKE_hi_E[i]-BDTGoutput_E[i])/(1.*test_data_trueKE_hi_E[i])
#   print("MC Energy: ", test_data_trueKE_hi_E[i]," Reco Energy: ",BDTGoutput_E[i]," DE/E[%]: ",Y[i])

df1 = pd.DataFrame(test_data_trueKE_hi_E,columns=['MuonEnergy'])
df2 = pd.DataFrame(BDTGoutput_E,columns=['RecoE'])
df_final = pd.concat([df1,df2],axis=1)
 
#-logical tests:
print("checking..."," df0.shape[0]: ",df1.shape[0]," len(y_predicted): ", len(BDTGoutput_E)) 
assert(df1.shape[0]==len(BDTGoutput_E))
assert(df_final.shape[0]==df2.shape[0])

#save results to .csv:  
df_final.to_csv("Ereco_results.csv", float_format = '%.3f')

nbins=np.arange(-100,100,2)
fig,ax0=plt.subplots(ncols=1, sharey=True)#, figsize=(8, 6))
cmap = sns.light_palette('b',as_cmap=True)
f=ax0.hist(np.array(Y), nbins, histtype='step', fill=True, color='gold',alpha=0.75)
ax0.set_xlim(-100.,100.)
ax0.set_xlabel('$\Delta E/E$ [%]')
ax0.set_ylabel('Number of Entries')
ax0.xaxis.set_label_coords(0.95, -0.08)
ax0.yaxis.set_label_coords(-0.1, 0.71)
title = "mean = %.2f, std = %.2f " % (np.array(Y).mean(), np.array(Y).std())
plt.title(title)
plt.savefig("DE_E.png")
