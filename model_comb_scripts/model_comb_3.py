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
from itertools import product

#memory and execution time measurement
import time
import tracemalloc



