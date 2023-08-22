import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
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
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "data_forRecoLength_beamlikeEvts.csv"
infile="/home/valas/data_for_trackLength_training.csv"
#infile="/home/valas/Documents/vtxreco-LAr40359940_0~700.csv"
#infile = "../data/data_forRecoLength_05202019.csv"
#infile = "/home/valas/Documents/vars_Ereco.csv"
#infile = "../data/data_forRecoLength_06082019CC0pi.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "../data_forRecoLength_04202019.csv"
#infile2 = "../data/data_forRecoLength_05202019.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

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
Dataset=np.array(pd.read_csv(filein))
np.random.shuffle(Dataset)#shuffling the data sample to avoid any bias in the training
print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)
print(rest)
print(features[:,2202])
print(features[:,2201])
print(labels)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
    model.add(CustomDropout(0.1, seed=150))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(CustomDropout(0.1, seed=150))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

# checkpoint
filepath="weights_bets_MCDropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=1, callbacks=callbacks_list, verbose=0)
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,10.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("keras_train_test.pdf")
#n, bins, patches = plt.hist(lambdamax, 50, density=1, facecolor='r', alpha=0.75)
#plt.savefig("TrueTrackLengthLambdamaxhist.pdf")
#plt.savefig("TrueTrackLengthhist.pdf")
#plt.scatter(lambdamax,labels)
#plt.savefig("ScatterplotTrueTrackLengthwithlambdamax.pdf")
