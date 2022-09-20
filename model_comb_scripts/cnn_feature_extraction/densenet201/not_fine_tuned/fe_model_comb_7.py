#import necessary libraries

##feature extraction
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
#from tensorflow import config
#from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from tensorflow.keras.backend import clear_session
from tensorflow import config
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.perform_feature_extraction import perform_feature_extraction
#from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.get_peak_gpu_mem_usage import get_peak_gpu_mem_usage

#memory and execution time measurement
import time
import tracemalloc

# #get GPU currently being used by you
# gpu_devices = config.list_physical_devices('GPU')

# #set this device to allow for memory growth
# for gpu in gpu_devices:
#   config.experimental.set_memory_growth(gpu, True)

############## Preparing dictionaries before function call ############################

#### paths to images and labels for each split ###
DATA_PATHS = {}

#training set
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/original/x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/y_train.npy'

#validation set
DATA_PATHS['validation_images'] = '/scratch/crwlia001/data/x_val.npy'
DATA_PATHS['validation_labels'] = '/scratch/crwlia001/data/y_val.npy'

#test set
DATA_PATHS['test_images'] = '/scratch/crwlia001/data/x_test.npy'
DATA_PATHS['test_labels'] = '/scratch/crwlia001/data/y_test.npy'

print('Loading training images/Fmaps and labels...\n')

x_train = np.load(DATA_PATHS['training_images'])
y_train = np.load(DATA_PATHS['training_labels'])

print('Training images/Fmaps and labels loaded!\n')

print('Loading validation images/Fmaps and labels...\n')

x_val = np.load(DATA_PATHS['validation_images'])
y_val = np.load(DATA_PATHS['validation_labels'])

print('Validation images/Fmaps and labels loaded!\n')

print('Loading test images/Fmaps and labels...\n')

x_test = np.load(DATA_PATHS['test_images'])
y_test = np.load(DATA_PATHS['test_labels'])

print('Test images/Fmaps and labels loaded!\n')

### feature extraction settings ###
FE_SETTINGS = {}
FE_SETTINGS['cnn_backbone_name'] = 'MobileNetV2'
FE_SETTINGS['candidate_layer_name'] = 'block_6_expand' #(28x28x192)
FE_SETTINGS['load_fine_tuned_model'] = False
FE_SETTINGS['best_dropout_rate'] = None
FE_SETTINGS['fine_tuned_weights_path'] = None

feature_extraction = {}

#Perform feature extraction
x_train, x_val, x_test, feature_extraction_memory_usage, feature_extraction_time = perform_feature_extraction(
    x_train = x_train,
    y_train = y_train,
    x_val = x_val,
    y_val = y_val,
    x_test = x_test,
    y_test = y_test,
    cnn_backbone_name = FE_SETTINGS['cnn_backbone_name'],
    candidate_layer_name = FE_SETTINGS['candidate_layer_name'],
    load_fine_tuned_model = FE_SETTINGS['load_fine_tuned_model'],
    best_dropout_rate = FE_SETTINGS['best_dropout_rate'],
    fine_tuned_weights_path = FE_SETTINGS['fine_tuned_weights_path']
    )

#add measurement results from feature extraction to relevant dictionaries
feature_extraction['peak_mem_usage'] = feature_extraction_memory_usage
feature_extraction['execution_time'] = feature_extraction_time

#feature extraction
np.save('/home/crwlia001/model_combination_results/combination_7/model_comb_7_feature_extraction.npy', feature_extraction)