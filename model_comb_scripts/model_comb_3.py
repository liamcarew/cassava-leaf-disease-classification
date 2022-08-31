# curated training set?: no
# augmentation?: no
# Feature Extraction?: yes
# Fine-tuning?: no
# CNN backbone: DenseNet201 (Backbone 1)
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
from model_comb_config import gcForestCS_model_config
from itertools import product

##memory and execution time measurement
import time
import tracemalloc

#specify training set feature map and associated label paths
X_TRAIN_PATH = '/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_x_train.npy'
Y_TRAIN_PATH = '/scratch/crwlia001/data/y_train.npy'

#specify validation set feature map and associated label paths
X_VAL_PATH = '/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_x_val.npy'
Y_VAL_PATH = '/scratch/crwlia001/data/y_val.npy'

#specify test set feature map and associated label paths
X_TEST_PATH = '/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_x_test.npy'
Y_TEST_PATH = '/scratch/crwlia001/data/y_test.npy'

#specify the different hyperparameters you wish to tune along with the associated values
COMBS_MGS = [(1, False), (2, True), (4, True)]
COMBS_CA = [(4, False), (8, True), (16, True)]
COMBS_POOLING_MGS = [False]

#Run hyperparameter gridsearch
gcForestCS_model_config(
    x_train_path = X_TRAIN_PATH,
    y_train_path = Y_TRAIN_PATH,
    x_val_path = X_VAL_PATH,
    y_val_path = Y_VAL_PATH,
    x_test_path = X_TEST_PATH,
    y_test_path = Y_TEST_PATH,
    combs_mgs = COMBS_MGS,
    combs_pooling_mgs = COMBS_POOLING_MGS,
    combs_ca = COMBS_CA,
    model_combination_num = 3)