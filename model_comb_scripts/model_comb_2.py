# Curated training set?: None
# Augmentation?: Yes
# Feature Extraction?: None
# CNN backbone: None (raw images)
# Classifier: gcForestCS

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

from itertools import product
import time
import tracemalloc

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

print('Loading balanced set of training images and associated labels...\n')

x_train = np.load('/scratch/crwlia001/x_train_raw_balanced.npy')
y_train = np.load('/scratch/crwlia001/y_train_raw_balanced.npy')

print('training images and associated labels loaded!\n')

print('Loading validation images and associated labels...\n')

x_val = np.load('/scratch/crwlia001/x_val.npy')
y_val = np.load('/scratch/crwlia001/y_val.npy')

print('validation images and associated labels loaded!\n')

print('Loading test images and associated labels...\n')

x_test = np.load('/scratch/crwlia001/x_test.npy')
y_test = np.load('/scratch/crwlia001/y_test.npy')

print('test images and associated labels loaded!\n')

################## Performing hyperparameter gridsearch #######################

################## Multi-grained scanning ##################################

print('Performing hyperparameter gridsearch...\n')

#specify the different hyperparameters you wish to tune along with the associated values
combs_mgs = [(1, False), (1, True), (2, True)]
combs_ca = [(4, False), (4, True), (8, True)]

# combs_mgs = [(1, False), (1, True)]
# combs_ca = [(1, False)]

#produce a list of all of the different hyperparameter combinations
hyperparameter_comb = [_ for _ in product(combs_mgs, combs_ca)]

## Reshape the training and validation inputs to format needed for multi-grained scanning (n_images, n_channels, width, height)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[3], x_val.shape[1], x_val.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2])

#create an empty dictionary which will be populated with the hyperparameter combination (key) along with the confusion matrix array (value)
conf_mats_val = {}
weighted_f1_scores_val = {}
oa_val = {}
conf_mats_test = {}
weighted_f1_scores_test = {}
oa_test = {}

#create an empty dictionary which will be populated with the hyperparameter combination (key) along with the peak RAM usage during training and prediction (value)
mem_usage_training = {}
mem_usage_prediction_val = {}
mem_usage_prediction_test = {}

#create empty dictionaries which will be populated with the hyperparameter combination (key) along with the execution times for training and prediction (value)
training_time = {}
prediction_time_val = {}
prediction_time_test = {}

for comb in hyperparameter_comb:

  #assign hyperparameters to variables
  n_estimators_mgs, tree_diversity_mgs = comb[0]
  n_estimators_ca, tree_diversity_ca = comb[1]

  print('Fitting gcForestCS model using the following hyperparameter settings:\nn_estimators: {}, tree_diversity_mgs: {}, n_estimators_ca: {}, tree_diversity_ca: {}\n'.format(n_estimators_mgs, tree_diversity_mgs, n_estimators_ca, tree_diversity_ca))

  #get model configuration
  config = build_gcforestCS(n_estimators_mgs = n_estimators_mgs,
                      	    tree_diversity_mgs = tree_diversity_mgs,
                            n_estimators_ca = n_estimators_ca,
                            tree_diversity_ca = tree_diversity_ca)

  #create a model instance using model configuration
  cnn_gc = GCForestCS(config)

  #initialise variables to monitor RAM usage and execution time during training
  tracemalloc.start()
  #tracemalloc.reset_peak()
  start_time_training = time.process_time()

  #fit model to training data
  cnn_gc.fit_transform(x_train, y_train)

  #terminate monitoring of RAM usage and execution time
  end_time_training = time.process_time()
  first_size, first_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  #determine training time and RAM usage
  training_exec_time = end_time_training - start_time_training
  #memory_usage_training = tracemalloc.get_traced_memory()

  #assign these values to respective dictionaries
  mem_usage_training[str(comb)] = first_peak / 1000000 #convert from bytes to megabytes
  training_time[str(comb)] = training_exec_time #in seconds

  ##repeat the above for predictions
  start_time_predictions_val = time.process_time()
  tracemalloc.start()

  #perform predictions
  y_val_pred = cnn_gc.predict(x_val)

  end_time_predictions_val = time.process_time()
  second_size, second_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  prediction_exec_time_val = end_time_predictions_val - start_time_predictions_val
  #memory_usage_prediction = tracemalloc.get_traced_memory()

  #assign these values to respective dictionaries
  mem_usage_prediction_val[str(comb)] = second_peak / 1000000 #convert from bytes to megabytes
  prediction_time_val[str(comb)] = prediction_exec_time_val #in seconds

  #produce confusion matrix
  cf_matrix_val = confusion_matrix(y_val, y_val_pred)
  conf_mats_val[str(comb)] = cf_matrix_val

  #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
  weighted_f1_scores_val[str(comb)] = round(f1_score(y_val, y_val_pred, average='weighted'), 3)

  #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
  oa_val[str(comb)] = round(accuracy_score(y_val, y_val_pred), 3)

  ##Repeat the above for test set
  start_time_predictions_test = time.process_time()
  tracemalloc.start()

  #perform predictions
  y_test_pred = cnn_gc.predict(x_test)

  end_time_predictions_test = time.process_time()
  third_size, third_peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  prediction_exec_time_test = end_time_predictions_test - start_time_predictions_test
  #memory_usage_prediction = tracemalloc.get_traced_memory()

  #assign these values to respective dictionaries
  mem_usage_prediction_test[str(comb)] = third_peak / 1000000 #convert from bytes to megabytes
  prediction_time_test[str(comb)] = prediction_exec_time_test #in seconds

  #produce confusion matrix
  cf_matrix_test = confusion_matrix(y_test, y_test_pred)
  conf_mats_test[str(comb)] = cf_matrix_test

  #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
  weighted_f1_scores_test[str(comb)] = round(f1_score(y_test, y_test_pred, average='weighted'), 3)

  #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
  oa_test[str(comb)] = round(accuracy_score(y_test, y_test_pred), 3)

  #calculate weighted f1-score (to account for class imbalance)
  #f1 = f1_score(y_val, y_val_pred, average='weighted')

  #add the result along with the hyperparameter selected to 'results_dict'
  #results[str(comb)] = f1

  #add the result along with the hyperparameter selected to 'results_dict'
  #conf_mats[str(comb)] = cf_matrix

################# Saving results from gridsearch ############################

#confusion matrices
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_conf_mats_val.npy', conf_mats_val)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_conf_mats_test.npy', conf_mats_test)

#weighted f1-scores
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_f1_val.npy', weighted_f1_scores_val)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_f1_test.npy', weighted_f1_scores_test)

#overall accuracy
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_oa_val.npy', oa_val)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_oa_test.npy', oa_test)

#peak memory usage
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_mem_usage_training.npy', mem_usage_training)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_mem_usage_prediction_val.npy', mem_usage_prediction_val)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_mem_usage_prediction_test.npy', mem_usage_prediction_test)

#training and prediction times
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_training_time.npy', training_time)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_prediction_time_val.npy', prediction_time_val)
np.save('/home/crwlia001/combination_2/model_comb_2_default_raw_prediction_time_test.npy', prediction_time_test)


#############################################################################


# # ################## Predictions ##############################################################

# print('\nPerforming predictions on validation set...\n')

# # # #feed validation data outputted from MGS into cascade forest classifier to produce predictions
# y_val_pred = gc.predict(x_val)

# # # #produce confusion matrix from 'y_pred' and 'y_val'
# cf_matrix = confusion_matrix(y_val, y_val_pred)

# #################### Saving confusion matrix #########################################

# #np.save('/scratch/crwlia001/cf_mat_nc_na_raw_img_gc_cs.npy', cf_matrix)

# ################### Calculating F1-score ########################################

# print('\nCalculated model metrics:\n')

# #calculate weighted f1-score (to account for class imbalance) and overall accuracy (OA)
# f1 = f1_score(y_val, y_val_pred, average='weighted')
# acc = accuracy_score(y_val, y_val_pred) * 100

# print("Validation weighted f1-score: {:.3f}".format(f1))
# print("Validation overall accuracy: {:.3f}%".format(acc))