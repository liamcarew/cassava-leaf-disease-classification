#import necessary libraries
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed
from tensorflow.keras.backend import clear_session
from tensorflow import config
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from itertools import product
import time
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.build_deep_learning_model import build_deep_learning_model
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.get_peak_gpu_mem_usage import get_peak_gpu_mem_usage
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.deep_learning_data_preparation import deep_learning_data_preparation
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.deep_learning_gridsearch import deep_learning_gridsearch
import numpy as np

############## Preparing for data preparation function call ############################

#### paths to images and labels for each split ###
DATA_PATHS = {}

#training set
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/original/x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/y_train.npy'

#validation set
DATA_PATHS['validation_images'] = '/scratch/crwlia001/data/x_val.npy'
DATA_PATHS['validation_labels'] = '/scratch/crwlia001/data/y_val.npy'

############ Perform data preparation ########################

training_data, validation_data, x_val, y_val = deep_learning_data_preparation(data_paths = DATA_PATHS, batch_size = 32)

############ Build deep learning model #######################

model = build_deep_learning_model(backbone = 'MobileNetV2', dropout_rate = 0.25, optimiser = 'sgd', learning_rate = 0.01, start_fine_tune_layer_name = 'block_3_depthwise')

#add early stopping as callback to model to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=75,
                               verbose=0,
                               restore_best_weights=True)

#specify number of epochs and batch size
EPOCHS = 100

#fit model to training data
set_seed(1) #for reproducibility
history = model.fit(training_data,
                    #batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = validation_data,
                    callbacks = [early_stopping],
                    shuffle=False,
                    verbose=0)

#save history as dictionary
np.save('/home/crwlia001/deep_learning_history_graphs/results/mobilenetv2_history.npy', history.history)