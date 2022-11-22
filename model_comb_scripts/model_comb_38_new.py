# curated training set?: yes
# augmentation?: no
# Feature Extraction?: Yes
# Fine-tuning?: Yes
# CNN backbone: DenseNet (Backbone 1)
# Candidate layer 3 ('Conv_1' #(7x7x1280))
# Pooling after MGS: yes
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
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/curated/curated_x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/training_set/curated/curated_y_train.npy'

#validation set
DATA_PATHS['validation_images'] = '/scratch/crwlia001/data/x_val.npy'
DATA_PATHS['validation_labels'] = '/scratch/crwlia001/data/y_val.npy'

#test set
DATA_PATHS['test_images'] = '/scratch/crwlia001/data/x_test.npy'
DATA_PATHS['test_labels'] = '/scratch/crwlia001/data/y_test.npy'

### hyperparameter settings in gridsearch ###
HYP_SETTINGS = {}
HYP_SETTINGS['combs_mgs'] = [50, 100]
HYP_SETTINGS['combs_pooling_mgs'] = [True]
HYP_SETTINGS['combs_ca'] = [50, 100]

### feature extraction settings ###
FE_SETTINGS = {}
FE_SETTINGS['cnn_backbone_name'] = 'MobileNetV2'
FE_SETTINGS['candidate_layer_name'] = 'Conv_1' #(7x7x1280)
FE_SETTINGS['load_fine_tuned_model'] = True
FE_SETTINGS['best_dropout_rate'] = 
FE_SETTINGS['fine_tuned_weights_path'] = 

################### Run Hyperparameter Gridsearch ####################################

gcForestCS_gridsearch(
    data_paths = DATA_PATHS,
    hyp_settings = HYP_SETTINGS,
    model_combination_num = 38,
    cnn_feature_extraction=True,
    feature_extraction_settings=FE_SETTINGS
    )