# curated training set?: no
# augmentation?: no
# CNN backbone: DenseNet201 (Backbone 1)
# Candidate layer: 28x28x128
# Classifier: Cascade forest

#import necessary libraries

import argparse
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
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
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.reshape_mgs_output import reshape_mgs_output
#from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.densenet import DenseNet201
#from tensorflow.keras.applications.efficientnet import EfficientNetB4
#from tensorflow_hub import KerasLayer
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
#from tensorflow.data import AUTOTUNE

from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcForest.gcforestCS import GCForestCS
#sys.path.append('./modelling/src/multi_grained_scanning/utils/gcForestCS/lib/gcForest')
#from gcforestCS import GCForestCS

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

print('Loading training feature maps and associated labels...\n')

x_train = np.load('/scratch/crwlia001/x_train_feature_maps.npy')
y_train = np.load('/scratch/crwlia001/y_train.npy')

print('training feature maps and associated labels loaded!\n')

print('Loading validation feature maps and associated labels...\n')

x_val = np.load('/scratch/crwlia001/x_val_feature_maps.npy')
y_val = np.load('/scratch/crwlia001/y_val.npy')

print('validation feature maps and associated labels loaded!\n')

################## Multi-grained scanning ##################################

print('Preparing for multi-grained scanning...\n')

## Produce architecture that will be used during multi-grained scanning
gc_cs_config = build_gcforestCS()

## Create a model instance based on configuration variable structure
cnn_gc_cs = GCForestCS(gc_cs_config)

## Set MGS model so that model will not be kept in memory (since RAM is bottleneck for algorithm)
cnn_gc_cs.set_keep_model_in_mem(flag=0)

## Reshape the training and validation inputs to format needed for multi-grained scanning (n_images, n_channels, width, height)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[3], x_val.shape[1], x_val.shape[2])

print('Fit gcForestCS model to training data...\n')

# ## Perform multi-grained scanning (MGS)
cnn_mgs = cnn_gc_cs.fit_transform(x_train, y_train, X_test = x_val, y_test = y_val)

print('gcForestCS model training complete!\n')

#################### Saving results from MGS #########################################

np.save('/scratch/crwlia001/mgs_results_densenet201_28x28x128.npy', cnn_mgs)

######################################################################################

# ## Produce np arrays of MGS outputs
# cnn_mgs_train = np.array(cnn_mgs_output[0])
# cnn_mgs_val = np.array(cnn_mgs_output[1])

# ## Reshape MGS outputs to that needed before feeding it into cascade forest classifier
# cnn_mgs_train = reshape_mgs_output(cnn_mgs_train)
# cnn_mgs_val = reshape_mgs_output(cnn_mgs_val)

# ################## Fitting cascade forest classifier ########################################

# #feed the feature vector produced from MGS into cascade forest classifier with default parameters
# model = CascadeForestClassifier(random_state=1)
# model.fit(cnn_mgs_train, y_train)

# ################## Predictions ##############################################################

# # #feed validation data outputted from MGS into cascade forest classifier to produce predictions
# y_val_pred = cnn_gc_cs.predict(x_val)

# # #produce confusion matrix from 'y_pred' and 'y_val'
# cf_matrix = confusion_matrix(y_val, y_val_pred)

# np.save('/scratch/crwlia001/cf_mat_nc_na_b1_cl1_gc_cs.npy', cf_matrix)

