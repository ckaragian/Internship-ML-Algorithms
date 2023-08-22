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
import lime
import lime.lime_tabular
import warnings
from lime import submodular_pick

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
infile2 = "/home/valas/Documents/ANNIE_ML/Internship-ML-Algorithms/vars_Ereco_pred.csv"
infile = "/home/valas/Documents/ANNIE_ML/Internship-ML-Algorithms/vars_Ereco_train.csv"
#infile2 = "../TrackLengthReconstruction/vars_Ereco_pred_06082019CC0pi.csv"
#----------------

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

E_threshold = 1200.
E_low=0
E_high=2000
div=100
bins = int((E_high-E_low)/div)
print('bins: ', bins)

print( "--- opening file with input variables!")
#--- events for training ---
filein = open(str(infile))
print("evts for training in: ",filein)
df00=pd.read_csv(filein)
df0=df00[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','recoTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel0=df0.loc[df0['trueKE'] < E_threshold]
dfsel=df0.dropna()
print("df0.head(): ", df0.head())

#print to check:
print("check training sample: \n",dfsel.head())
#   print(dfsel.iloc[5:10,0:5])
#check fr NaN values:
print("The dimensions of training sample ",dfsel.shape)
assert(dfsel.isnull().any().any()==False)

#--- normalisation-training sample:
#dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR']/152.4, dfsel['recoDWallZ']/198., dfsel['totalLAPPDs']/1000., dfsel['totalPMTs']/1000., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['recoTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR'], dfsel['recoDWallZ'], dfsel['totalLAPPDs']/200., dfsel['totalPMTs']/200., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
print("check normalisation: ", dfsel_n.head())

MRDTrackLength=dfsel_n['recoTrackLengthInMrd']
i=0
a=[]
for y in MRDTrackLength:
   if y<0:
     print("MRDTrackLength:",y,"Event:",i)
     a.append(i)
   i=i+1
dfsel_n1=dfsel_n.drop(dfsel_n.index[a])
dfsel1=dfsel.drop(dfsel.index[a])

#--- events for predicting ---
filein2 = open(str(infile2))
print(filein2)
df00b = pd.read_csv(filein2)
df0b=df00b[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','recoTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel_pred=df0b.loc[df0b['trueKE'] < E_threshold]
dfsel_pred=df0b
#print to check:
print("check predicting sample: ",dfsel_pred.shape," ",dfsel_pred.head())
#   print(dfsel_pred.iloc[5:10,0:5])
#check fr NaN values:
assert(dfsel_pred.isnull().any().any()==False)

#--- normalisation-sample for prediction:
#dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd']/200., dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR']/152.4, dfsel_pred['recoDWallZ']/198., dfsel_pred['totalLAPPDs']/1000., dfsel_pred['totalPMTs']/1000., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T
dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['recoTrackLengthInMrd']/200., dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR'], dfsel_pred['recoDWallZ'], dfsel_pred['totalLAPPDs']/200., dfsel_pred['totalPMTs']/200., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T

#prepare events for predicting 

MRDTrackLength=dfsel_pred_n['recoTrackLengthInMrd']
i=0
a=[]
for y in MRDTrackLength:
   if y<0:
     print("MRDTrackLength:",y,"Event:",i)
     a.append(i)
   i=i+1
dfsel_pred_n1=dfsel_pred_n.drop(dfsel_pred_n.index[a])
dfsel_pred1=dfsel_pred.drop(dfsel_pred.index[a])

evts_to_predict_n= np.array(dfsel_pred_n1[['DNNRecoLength','recoTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs', 'vtxX','vtxY','vtxZ']])
#evts_to_predict_n1= np.delete(evts_to_predict_n, 1, axis=1)
test_data_trueKE_hi_E = np.array(dfsel_pred1[['trueKE']])

#--- prepare training & test sample for BDT:
arr_hi_E0 = np.array(dfsel_n1[['DNNRecoLength','recoTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']])
#arr_hi_E1 = np.delete(arr_hi_E0, 1, axis=1)
arr3_hi_E0 = np.array(dfsel1[['trueKE']])
 
#---- random split of events ----
rnd_indices = np.random.rand(len(arr_hi_E0)) < 1. #< 0.50
print(rnd_indices[0:5])
#--- select events for training/test:
arr_hi_E0B = arr_hi_E0[rnd_indices]
print(arr_hi_E0B[0:5])
arr2_hi_E_n = arr_hi_E0B #.reshape(arr_hi_E0B.shape + (-1,))
arr3_hi_E = arr3_hi_E0[rnd_indices]
##--- select events for prediction: -- in future we need to replace this with data sample!
#evts_to_predict = arr_hi_E0[~rnd_indices]
#evts_to_predict_n = evts_to_predict #.reshape(evts_to_predict.shape + (-1,))
#test_data_trueKE_hi_E = arr3_hi_E0[~rnd_indices]

#printing..
print('events for training: ',len(arr3_hi_E)) #,' events for predicting: ',len(test_data_trueKE_hi_E)) 
print('initial train shape: ',arr3_hi_E.shape) #," predict: ",test_data_trueKE_hi_E.shape)

########### BDTG ############
n_estimators=500
params = {'n_estimators':n_estimators, 'max_depth': 50,
          'learning_rate': 0.025, 'loss': 'absolute_error'} 

print("arr2_hi_E_n.shape: ",arr2_hi_E_n.shape)
#--- select 70% of sample for training and 30% for testing:
offset = int(arr2_hi_E_n.shape[0] * 0.7) 
arr2_hi_E_train, arr3_hi_E_train = arr2_hi_E_n[:offset], arr3_hi_E[:offset].reshape(-1)  # train sample
arr2_hi_E_test, arr3_hi_E_test   = arr2_hi_E_n[offset:], arr3_hi_E[offset:].reshape(-1)  # test sample
 
print("train shape: ", arr2_hi_E_train.shape," label: ",arr3_hi_E_train.shape)
print("test shape: ", arr2_hi_E_test.shape," label: ",arr3_hi_E_test.shape)
    
print("training BDTG...")
net_hi_E = ensemble.GradientBoostingRegressor(**params)
model = net_hi_E.fit(arr2_hi_E_train, arr3_hi_E_train)
net_hi_E

#Define lime explainer
features_names=[]
for col in dfsel_n.columns:
    features_names.append(col)
explainer = lime.lime_tabular.LimeTabularExplainer(arr2_hi_E_train, feature_names=features_names, class_names=['trueKE'], verbose=True, mode='regression')

# load the model from disk
filename = 'finalized_BDTmodel_forMuonEnergy.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

#predicting...
print("events for energy reco: ", len(evts_to_predict_n)) 
BDTGoutput_E = loaded_model.predict(evts_to_predict_n)

# Explain a prediction
exp = explainer.explain_instance(evts_to_predict_n[100], loaded_model.predict, num_features=5)
exp.save_to_file('explanationBDT100.html')

print("Predicted value for event 100:", BDTGoutput_E[100])

sp_obj = submodular_pick.SubmodularPick(explainer, evts_to_predict_n, loaded_model.predict, sample_size=20, num_features=5, num_exps_desired=5)

[exp.save_to_file('explanationBDT'+str(i)+'.html') for i, exp in enumerate(sp_obj.sp_explanations)]

Y=[0 for j in range (0,len(test_data_trueKE_hi_E))]
for i in range(len(test_data_trueKE_hi_E)):
    Y[i] = 100.*(test_data_trueKE_hi_E[i]-BDTGoutput_E[i])/(1.*test_data_trueKE_hi_E[i])
#   print("MC Energy: ", test_data_trueKE_hi_E[i]," Reco Energy: ",BDTGoutput_E[i]," DE/E[%]: ",Y[i])

df1 = pd.DataFrame(test_data_trueKE_hi_E,columns=['MuonEnergy'])
df2 = pd.DataFrame(BDTGoutput_E,columns=['RecoE'])
df3 = pd.DataFrame(evts_to_predict_n[:,0], columns=['DNNRecoLength'])
df4 = pd.DataFrame(evts_to_predict_n[:,1], columns=['TrueTrackLengthInMrd'])
df_final = pd.concat([df1,df2],axis=1)
df_final1 = pd.concat([df3,df4], axis=1)
df_final2= pd.concat([df_final,df_final1], axis=1)
#-logical tests:
print("checking..."," df0.shape[0]: ",df1.shape[0]," len(y_predicted): ", len(BDTGoutput_E))
print("checking..."," df_final.shape[0]: ",df_final.shape[0]," df_final1.shape[0]: ", df_final1.shape[0], " df_final2.shape[0]: ", df_final2.shape[0])
assert(df1.shape[0]==len(BDTGoutput_E))
assert(df_final.shape[0]==df2.shape[0])

#save results to .csv:  
df_final2.to_csv("Ereco_results.csv", float_format = '%.3f')

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
