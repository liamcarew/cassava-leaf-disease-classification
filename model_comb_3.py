# curated training set?: no
# augmentation?: no
# Feature Extraction?: No
# Fine-tuning?: Yes
# CNN backbone: DenseNet201 (Backbone 1)
# Classifier: FCN

#import necessary libraries
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn import utils
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.build_deep_learning_model import build_deep_learning_model
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed
from itertools import product
import time
import tracemalloc

###################### Importing and Preparing Data ###################################

print('Loading training images and associated labels...\n')

x_train = np.load('/scratch/crwlia001/x_train.npy')
y_train = np.load('/scratch/crwlia001/y_train.npy')

print('training images and associated labels loaded!\n')

print('Loading validation images and associated labels...\n')

x_val = np.load('/scratch/crwlia001/x_val.npy')
y_val = np.load('/scratch/crwlia001/y_val.npy')

print('validation images and associated labels loaded!\n')

BATCH_SIZE = 32

#first, let's convert 'x_train' and 'y_train' into tensorflow dataset object, applying functions that will optimise training
training_data = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10).batch(batch_size = BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)

#do the same for the validation set
validation_data = Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size = BATCH_SIZE)

################## Hyperparameter gridsearch #######################

#create a parameter grid of the different values to use during gridsearch
dropout_rate = [0.25, 0.5, 0.75]
optimiser = ['adam', 'sgd']
learning_rate = [0.0001, 0.001, 0.01]

#produce a list of all of the different hyperparameter combinations
hyperparameter_comb = [_ for _ in product(dropout_rate, optimiser, learning_rate)]

#let's define dictionaries that will be needed

#confusion matrices
conf_mats_val = {}
f1_val = {}
overall_acc_val = {}

#memory usage
mem_usage_training = {}
mem_usage_prediction_val = {}

#execution time
training_time = {}
prediction_time_val = {}

for comb in hyperparameter_comb:

  #save each of the hyperparameters in this combination as separate variables
  dropout_rate, optimiser, learning_rate = comb

  #create deep learning model instance with these hyperparameters
  model = build_deep_learning_model(backbone = 'densenet201',
                                    dropout_rate = dropout_rate,
                                    optimiser = optimiser,
                                    learning_rate = learning_rate,
                                    layer_num = 52)

  #add early stopping as callback to model to prevent overfitting
  early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=15,
                                 verbose=1,
                                 restore_best_weights=True)

  #specify number of epochs and batch size
  EPOCHS = 250
  BATCH_SIZE = 32        

  #initialise variables to monitor RAM usage and execution time during training
  tracemalloc.start()
  start_time_training = time.process_time()

  #fit model to training data
  set_seed(1) #for reproducibility
  model.fit(training_data,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            validation_data = validation_data,
            callbacks = [early_stopping],
            shuffle=True
            )

  #terminate monitoring of RAM usage and execution time
  end_time_training = time.process_time()
  first_size, first_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  #determine training time
  training_exec_time = end_time_training - start_time_training

  #assign peak RAM usage and execution time to respective dictionaries
  mem_usage_training[str(comb)] = first_peak / 1000000 #convert from bytes to megabytes
  training_time[str(comb)] = training_exec_time #in seconds

  ##perform predictions on validation set
  start_time_predictions_val = time.process_time()
  tracemalloc.start()

  y_val_pred = model.predict_classes(x_val)

  end_time_predictions_val = time.process_time()
  second_size, second_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  prediction_exec_time_val = end_time_predictions_val - start_time_predictions_val

  #assign peak memory usage and prediction times to respective dictionaries
  mem_usage_prediction_val[str(comb)] = second_peak / 1000000 #convert from bytes to megabytes
  prediction_time_val[str(comb)] = prediction_exec_time_val #in seconds

  #produce confusion matrix
  cf_matrix_val = confusion_matrix(y_val, y_val_pred)
  conf_mats_val[str(comb)] = cf_matrix_val

  #calculate weighted f1-score and overall accuracy and add these to their respective dictionaries
  f1 = f1_score(y_val, y_val_pred, average='weighted')
  oa = accuracy_score(y_val, y_val_pred)

  f1_val[str(comb)] = round(f1, 2)
  overall_acc_val[str(comb)] = round(oa, 2)