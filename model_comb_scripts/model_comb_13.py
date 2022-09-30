# curated training set?: no
# augmentation?: no
# Feature Extraction?: No
# Fine-tuning?: Yes
# CNN backbone: MobileNetV2 (Backbone 1)
# Classifier: FCN

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

############## configure GPU to allow for memory usage measurement #####################

#get GPU currently being used by you
gpu_devices = config.list_physical_devices('GPU')

#set this device to allow for memory growth
for gpu in gpu_devices:
  config.experimental.set_memory_growth(gpu, True)

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

############ Prepare gridsearch #######################

#create a parameter grid of the different values to use during gridsearch
dropout_rate = [0.25, 0.5, 0.75]
optimiser = ['adam', 'sgd']
learning_rate = [0.0001, 0.001, 0.01]

#produce a list of all of the different hyperparameter combinations
hyperparameter_comb = [_ for _ in product(dropout_rate, optimiser, learning_rate)]

############ Perform deep learning gridsearch ################

deep_learning_gridsearch(
  hyperparameter_combinations = hyperparameter_comb,
  model_combination_num = 13,
  backbone = 'MobileNetV2',
  training_data = training_data,
  validation_data = validation_data,
  x_val = x_val,
  y_val = y_val,
  num_epochs = 100,
  random_state = 1,
  start_fine_tune_layer_name = 'block_3_depthwise',
  es_patience = 75,
  gpu_devices = gpu_devices)