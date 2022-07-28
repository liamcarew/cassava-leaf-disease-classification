# curated training set?: no
# augmentation?: no
# for which CNN backbone?: None
# Raw images?: yes (224x224x3) - input dimensions for DenseNet201
# Candidate layer?: None
# Classifier: XGBoost

#import necessary libraries

import argparse
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
#import tensorflow as tf
#import tensorflow_datasets as tfds
#from deepforest import CascadeForestClassifier
import random
from sklearn import utils
#from image_pre_processing.utils.data_generators_with_no_aug import image_preprocessing
#from image_pre_processing.utils.convert_to_np_array import convert_to_np_array
##from modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest import gcforestCS
#from modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.build_gcForestCS import build_gcforestCS
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.reshape_inputs import reshape_inputs
#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.densenet import DenseNet201
#from tensorflow.keras.applications.efficientnet import EfficientNetB4
#from tensorflow_hub import KerasLayer
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
#from tensorflow.data import AUTOTUNE

from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest.gcforestCS import GCForestCS
#sys.path.append('./modelling/src/multi_grained_scanning/utils/gcForestCS/lib/gcForest')
#from gcforest import GCForest

import xgboost as xgb

###################### Importing Data ###################################

# print('Importing data...\n')

# ##read in cassava dataset
# dataset = tfds.load('cassava')

# print('Data import complete!\n')

##################### Image Pre-processing #############################

# print('Performing image pre-processing...\n')

# ##apply image pre-processing to each data split
# training_data = dataset['train'].map(image_preprocessing)
# validation_data = dataset['validation'].map(image_preprocessing)
# #test_data = dataset['test'].map(image_preprocessing)

# ##produce np arrays of the training set and validation set class labels respectively
# y_train = convert_to_np_array(dataset = training_data, split_name = 'training')
# y_val = convert_to_np_array(dataset = validation_data, split_name = 'validation')

# ##prepare training and validation image sets for processing in feature extractor

# #define global variables
# #AUTOTUNE = tf.data.AUTOTUNE
# batch_size = 16

# #prepare training and validation image sets
# x_train = training_data.cache().batch(batch_size)
# x_val = validation_data.cache().batch(batch_size)

# print('Image Pre-processing complete!\n')

################## CNN Feature Extraction ################################

# print('Building CNN feature extractor...\n')

# ##Build pre-trained CNN feature extractor using model run specifications
# cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = 'DenseNet201', output_layer_name = 'conv3_block1_1_conv') #28x28x128

# print('Performing feature extraction...\n')

# ##Fit training data and validation data to feature extractor to produce feature maps using 'imagenet' weights
# x_train = cnn_feature_extractor.predict(x = x_train,
#                                         batch_size = batch_size,
#                                         verbose = 1)
# x_val = cnn_feature_extractor.predict(x = x_val, batch_size = batch_size, verbose = 1)

# ##Convert both to np arrays
# x_train = np.array(x_train)
# x_val = np.array(x_val)

# print('Feature extraction complete!\n')

print('Loading training images and associated labels...\n')

x_train = np.load('/scratch/crwlia001/x_train.npy')
y_train = np.load('/scratch/crwlia001/y_train.npy')

print('training images and associated labels loaded!\n')

print('Loading validation images and associated labels...\n')

x_val = np.load('/scratch/crwlia001/x_val.npy')
y_val = np.load('/scratch/crwlia001/y_val.npy')

print('validation images and associated labels loaded!\n')

################## Reshaping input for XGBoost #############################

print('Preparing images for XGBoost...\n')

#Reshape 4D images arrays (n_obs, height, width, channels) into 2D vector (n_obs, n_elements)
x_train = reshape_inputs(x_train)
x_val = reshape_inputs(x_val)

################## Fitting raw data to XGBoost model #######################

print('\nPerforming hyperparameter grid search...\n')

#Specific the values for hyperparameters during gridsearch
param_grid = {
    "n_estimators": [2,4],
    "max_depth": [2,4]
    }

#create xgb model instance
xgb_model = xgb.XGBClassifier()

#specify how to perform gridsearch (5-fold CV)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring = 'f1_weighted',
    n_jobs = 1,
    cv = 5,
    verbose=True
)

#Perform gridsearch
grid_search.fit(x_train, y_train)

#get best model hyperparameter combination
grid_search.best_estimator_


# ################## Predictions ##############################################################

print('\nPerforming predictions on validation set...\n')

# # #feed validation data outputted from MGS into cascade forest classifier to produce predictions
y_val_pred = gc.predict(x_val)

# # #produce confusion matrix from 'y_pred' and 'y_val'
cf_matrix = confusion_matrix(y_val, y_val_pred)

#################### Saving confusion matrix #########################################

#np.save('/scratch/crwlia001/cf_mat_nc_na_raw_img_gc_cs.npy', cf_matrix)

################### Calculating F1-score ########################################

print('\nCalculated model metrics:\n')

#calculate weighted f1-score (to account for class imbalance) and overall accuracy (OA)
f1 = f1_score(y_val, y_val_pred, average='weighted')
acc = accuracy_score(y_val, y_val_pred) * 100

print("Validation weighted f1-score: {:.3f}".format(f1))
print("Validation overall accuracy: {:.3f}%".format(acc))