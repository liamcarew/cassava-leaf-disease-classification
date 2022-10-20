# curated training set?: no
# augmentation?: yes
# Feature Extraction?: Yes
# Fine-tuning?: Yes
# CNN backbone: DenseNet (Backbone 1)
# Candidate layer 1 ('pool3_conv' (28x28x256))
# Classifier: gcForestCS

#import necessary libraries

##gcForestCS
import argparse
import numpy as np
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import random
from sklearn import utils
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.build_gcForestCS import build_gcforestCS
#from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.reshape_inputs import reshape_inputs
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest.gcforestCS import GCForestCS
from model_comb_config import gcForestCS_gridsearch
from itertools import product

##memory and execution time measurement
import time
import tracemalloc

############## Preparing dictionaries before function call ############################

#### paths to images and labels for each split ###
DATA_PATHS = {}

#training set
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy'

#validation set
DATA_PATHS['validation_images'] = '/scratch/crwlia001/data/x_val.npy'
DATA_PATHS['validation_labels'] = '/scratch/crwlia001/data/y_val.npy'

#test set
DATA_PATHS['test_images'] = '/scratch/crwlia001/data/x_test.npy'
DATA_PATHS['test_labels'] = '/scratch/crwlia001/data/y_test.npy'

### hyperparameter settings in gridsearch ###
HYP_SETTINGS = {}
HYP_SETTINGS['combs_mgs'] = [250, 500, 1000]
HYP_SETTINGS['combs_pooling_mgs'] = [False]
HYP_SETTINGS['combs_ca'] = [250, 500, 1000]

### feature extraction settings ###
FE_SETTINGS = {}
FE_SETTINGS['cnn_backbone_name'] = 'DenseNet201'
FE_SETTINGS['candidate_layer_name'] = 'pool3_conv' #(28x28x256)
FE_SETTINGS['load_fine_tuned_model'] = True
FE_SETTINGS['best_dropout_rate'] = 0.75
FE_SETTINGS['fine_tuned_weights_path'] = '/scratch/crwlia001/fine_tuned_model_weights/DenseNet201/model_comb_12_0.75_adam_0.0001.h5'

################### Run Hyperparameter Gridsearch ####################################

gcForestCS_gridsearch(
    data_paths = DATA_PATHS,
    hyp_settings = HYP_SETTINGS,
    model_combination_num = 17,
    cnn_feature_extraction=True,
    feature_extraction_settings=FE_SETTINGS
    )