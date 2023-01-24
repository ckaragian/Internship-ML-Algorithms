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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "../data_forRecoLength_04202019.csv"
#infile = "../data/data_forRecoLength_05202019.csv"
infile = "data_forRecoLength_beamlikeEvts.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "../data_forRecoLength_04202019.csv"
#infile2 = "../data/data_forRecoLength_05202019.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein, index_col=0))
print(Dataset)
np.random.shuffle(Dataset)
print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)

#--- events for predicting
#filein2 = open(str(infile2))
#print("events for prediction in: ",filein2)
#Dataset2 = np.array(pd.read_csv(filein2))
#features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
#print( "lambdamax2 ", lambdamax2[:2], labels[:2])
#print(features2[0])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]
#test_x = features2[1000:]
#test_y = labels2[1000:]
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def create_model(optimizer='Adamax', init_mode='normal', activation='relu', neurons1=25, neurons2=25):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_dim=2203, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(neurons2, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation=activation))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

#--- define the grid search parameters: comment out one option at a time to reduce CPU time!
#1) tune batch_size, epochs
batch_size = [1, 2, 5]
#epochs = [10, 50, 100, 500]
epochs = [10, 12, 15]
param_grid = dict(batch_size=batch_size, epochs=epochs)

#2) tune the training optimisation algorithm, the weight initialisation, 
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(optimizer=optimizer, init_mode=init_mode)

#3) tune the activation function, the number of neurons for the 1st layer
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
neurons = [25, 50, 70, 90, 100]
#param_grid = dict(neurons1=neurons,activation=activation)

#4) tune the number of neurons for the 2nd layer
neurons2 = [5, 10, 15, 20, 25, 30]
#param_grid = dict(neurons2=neurons2)

# search the grid parameters:
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(train_x, train_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

## checkpoint
#filepath="weights_bets.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
#callbacks_list = [checkpoint]
## Fit the model
#model.fit(train_x, train_y, validation_split=0.33, epochs=50, batch_size=2, callbacks=callbacks_list, verbose=0)

#-----------------------------
