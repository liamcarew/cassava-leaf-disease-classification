# curated training set?: no
# augmentation?: yes
# Feature Extraction?: yes
# Fine-tuning?: no
# CNN backbone: MobileNetV2 (Backbone 2)
# Candidate layer 3 ('Conv_1' #(7x7x1280))
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

#### paths to images and labels for each split ###
DATA_PATHS = {}

#specify training set feature map and associated label paths
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy'

#specify validation set feature map and associated label paths
DATA_PATHS['validation_images'] = '/scratch/crwlia001/data/x_val.npy'
DATA_PATHS['validation_labels'] = '/scratch/crwlia001/data/y_val.npy'

#specify test set feature map and associated label paths
DATA_PATHS['test_images'] = '/scratch/crwlia001/data/x_test.npy'
DATA_PATHS['test_labels'] = '/scratch/crwlia001/data/y_test.npy'

### hyperparameter settings in gridsearch ###
HYP_SETTINGS = {}
HYP_SETTINGS['combs_mgs'] = [50, 100]
HYP_SETTINGS['combs_pooling_mgs'] = [False]
HYP_SETTINGS['combs_ca'] = [50, 100]

### feature extraction settings ###
FE_SETTINGS = {}
FE_SETTINGS['cnn_backbone_name'] = 'MobileNetV2'
FE_SETTINGS['candidate_layer_name'] = 'Conv_1' #(7x7x1280)
FE_SETTINGS['load_fine_tuned_model'] = False
FE_SETTINGS['best_dropout_rate'] = None
FE_SETTINGS['fine_tuned_weights_path'] = None

################### Run Hyperparameter Gridsearch ####################################

gcForestCS_gridsearch(
    data_paths = DATA_PATHS,
    hyp_settings = HYP_SETTINGS,
    model_combination_num = 14,
    cnn_feature_extraction=True,
    feature_extraction_settings=FE_SETTINGS
    )