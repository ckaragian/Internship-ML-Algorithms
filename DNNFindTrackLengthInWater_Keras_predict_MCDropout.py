import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import lime
import lime.lime_tabular
import keras.backend as K
from scipy.stats import norm
import ROOT

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "data_forRecoLength_beamlikeEvts.csv"
infile="/home/valas/data_for_trackLength_training.csv"
#infile="/home/valas/Documents/vtxreco-LAr40359940_0~700.csv"
#infile = "../data/data_forRecoLength_05202019.csv"
#infile = "/home/valas/Documents/vars_ErecoMRDALL.csv"
#infile = "../data/data_forRecoLength_06082019CC0pi.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "data_forRecoLength_beamlikeEvts.csv"
infile3="/home/valas/Documents/ANNIE_ML/Internship-ML-Algorithms/vars_Ereco.csv"
infile2="/home/valas/data_for_trackLength_training.csv"
#infile2="/home/valas/Documents/vtxreco-LAr40359940_0~700.csv"
#infile2 = "../data/data_forRecoLength_05202019.csv"
#infile2 = "/home/valas/Documents/vars_ErecoMRDALL.csv"
#infile2 = "../data/data_forRecoLength_06082019CC0pi.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#
"""
class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    def predict(self,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result
"""
class CustomDropout(keras.layers.Dropout):
    def __init(self,rate,**kwargs):
        super().__init__(**kwargs)
        self.rate=rate
    def call(self,inputs):
        return tf.nn.dropout(inputs,rate=self.rate)

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset = np.array(pd.read_csv(filein))
#Dataset1=np.delete(Dataset,obj=1398,axis=0)
np.random.shuffle(Dataset)
print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)

#--- events for predicting
filein2 = open(str(infile2))
print("events for prediction in: ",filein2)
Dataset2 = np.array(pd.read_csv(filein2))
#Dataset22=np.delete(Dataset2,obj=1398,axis=0)
np.random.seed(seed)
np.random.shuffle(Dataset2)
print(Dataset2)
features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
print( "lambdamax2 ", lambdamax2[:2], labels[:2])
print(features2[0])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]
test_x = features2[2000:]
test_y = labels2[2000:]
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

# create model
model = Sequential()
model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
model.add(CustomDropout(0.1,seed=150))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(CustomDropout(0.1,seed=150))
model.add(Dense(1, kernel_initializer='normal', activation='relu'))

# load weights
model.load_weights("weights_bets_MCDropout.hdf5")

# Compile model
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
print("Created model and loaded weights from file")

## Predict.
print('predicting...')
x_transformed = scaler.transform(test_x)
predictions=[]
for i in range(200):
  y_predicted = model.predict(x_transformed)
  for j in range(y_predicted.shape[0]):
     predictions.append(y_predicted[j,:])
print(predictions[0])
print(predictions[1])
predictions=np.array(predictions).reshape(y_predicted.shape[0],200)
print(predictions.shape)
print(predictions[0])
predictions_mean=np.mean(predictions,axis=1)
predictions_std=np.std(predictions,axis=1)
predictions_uncertainty=np.divide(predictions_std,predictions_mean)*100


canvas=ROOT.TCanvas()
canvas.cd()
hist=ROOT.TH1F("hist", "Uncertainty", 20, 0., 50.)
for i in range(len(predictions_uncertainty)):
      hist.Fill(predictions_uncertainty[i])
hist.Draw()
canvas.Draw()
canvas.SaveAs("Uncertainty.png")

canvas1=ROOT.TCanvas()
canvas1.cd()
hist1=ROOT.TH1F("hist1", "Distribution", 100,min(predictions_mean),max(predictions_mean))
for i in range(len(predictions_mean)):
   hist1.Fill(predictions_mean[i])
f = ROOT.TF1( 'f', 'gaus',min(predictions_mean),  195)
hist1.Fit(f,"R")
hist1.Draw()
canvas1.Draw()
canvas1.SaveAs("Distribution.png")

filein3 = open(str(infile3))
df=pd.read_csv(filein3)
DNNRecoLength=df['DNNRecoLength']
DNNRecoLength=np.array(DNNRecoLength)

print(DNNRecoLength.shape,predictions_mean.shape)

deviation=(DNNRecoLength-predictions_mean)

rel_dev=np.divide(deviation,predictions_mean)

canvas2=ROOT.TCanvas()
canvas2.cd()
hist2=ROOT.TH1F("hist2", "Deviation", 100,-10,10)
for i in range(len(rel_dev)):
   hist2.Fill(rel_dev[i])
#f = ROOT.TF1( 'f', 'gaus',min(predictions_mean),  195)
#hist1.Fit(f,"R")
hist2.Draw()
canvas2.Draw()
canvas2.SaveAs("Deviation.png")
