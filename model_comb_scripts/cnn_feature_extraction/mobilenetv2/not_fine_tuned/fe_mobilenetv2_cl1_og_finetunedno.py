#import necessary libraries

##feature extraction
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow import config
#from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.data import Dataset, AUTOTUNE
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.perform_feature_extraction import perform_feature_extraction

#memory and execution time measurement
import time
import tracemalloc

#Perform feature extraction
perform_feature_extraction(
    training_set_type = 'original',
    cnn_backbone_name = 'MobileNetV2',
    candidate_layer_name = 'block_6_expand', #(28x28x192)
    candidate_layer_num = 1,
    model_combination = 7)