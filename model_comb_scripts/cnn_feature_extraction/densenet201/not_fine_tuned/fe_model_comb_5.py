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
DATA_PATHS['training_images'] = '/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy'
DATA_PATHS['training_labels'] = '/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy'

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
FE_SETTINGS['cnn_backbone_name'] = 'DenseNet201'
FE_SETTINGS['candidate_layer_name'] = 'pool3_conv' #(28x28x256)
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
np.save('/home/crwlia001/model_combination_results/combination_5/model_comb_5_feature_extraction.npy', feature_extraction)

################### Importing Data #############################

# #training set
# print('Loading training images and associated labels...\n')

# x_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy')
# y_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy')

# print('training images and associated labels loaded!\n')

# #validation set
# print('Loading validation images and associated labels...\n')

# x_val = np.load('/scratch/crwlia001/data/x_val.npy')
# y_val = np.load('/scratch/crwlia001/data/y_val.npy')

# print('validation images and associated labels loaded!\n')

# #test set
# print('Loading test images and associated labels...\n')

# x_test = np.load('/scratch/crwlia001/data/x_test.npy')
# y_test = np.load('/scratch/crwlia001/data/y_test.npy')

# print('test images and associated labels loaded!\n')

# ################## CNN Feature Extraction ################################

# print('Building CNN feature extractor...\n')

# #Build pre-trained CNN feature extractor using model run specifications
# cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = 'DenseNet201', output_layer_name = 'pool3_conv') #(28x28x256)

# print('Performing feature extraction...\n')

# #Extract feature maps from pre-trained CNN using 'imagenet' weights
# BATCH_SIZE = 32

# feature_extraction_memory_usage = {}
# feature_extraction_time = {}

# #prepare data before running
# BATCH_SIZE = 32
# AUTOTUNE = data.AUTOTUNE

# training_data = data.Dataset.from_tensor_slices((x_train, y_train)).cache().batch(batch_size = BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# validation_data = data.Dataset.from_tensor_slices((x_val, y_val)).cache().batch(batch_size = BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# testing_data = data.Dataset.from_tensor_slices((x_test, y_test)).cache().batch(batch_size = BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# ##training set

# #initialise memory usage and execution time measurement variables
# #tracemalloc.start()
# start_time_training = time.process_time()

# x_train = cnn_feature_extractor.predict(x = training_data, verbose = 1)

# #terminate monitoring of RAM usage and execution time
# end_time_training = time.process_time()
# #first_size, first_peak = tracemalloc.get_traced_memory()
# #tracemalloc.stop()

# #determine execution time
# training_exec_time = end_time_training - start_time_training

# #save memory usage and execution time to respective dictionaries
# #feature_extraction_memory_usage['train'] = first_peak / 1000000 #convert from bytes to megabytes
# feature_extraction_time['train'] = training_exec_time #in seconds

# ##validation set

# #initialise memory usage and execution time measurement variables
# #tracemalloc.start()
# start_time_val = time.process_time()

# x_val = cnn_feature_extractor.predict(x = validation_data, verbose = 1)

# #terminate monitoring of RAM usage and execution time
# end_time_val = time.process_time()
# #second_size, second_peak = tracemalloc.get_traced_memory()
# #tracemalloc.stop()

# #determine execution time
# val_exec_time = end_time_val - start_time_val

# #save memory usage and execution time to respective dictionaries
# #feature_extraction_memory_usage['val'] = second_peak / 1000000 #convert from bytes to megabytes
# feature_extraction_time['val'] = val_exec_time #in seconds

# ##test set

# #initialise memory usage and execution time measurement variables
# #tracemalloc.start()
# start_time_test = time.process_time()

# x_test = cnn_feature_extractor.predict(x = testing_data, verbose = 1)

# #terminate monitoring of RAM usage and execution time
# end_time_test = time.process_time()
# #third_size, third_peak = tracemalloc.get_traced_memory()
# #tracemalloc.stop()

# #determine execution time
# test_exec_time = end_time_test - start_time_test

# #save memory usage and execution time to respective dictionaries
# #feature_extraction_memory_usage['test'] = third_peak / 1000000 #convert from bytes to megabytes
# feature_extraction_time['test'] = test_exec_time #in seconds

# #Convert all to np arrays
# x_train = np.array(x_train)
# x_val = np.array(x_val)
# x_test = np.array(x_test)

# #shuffle training

# print('Feature extraction complete!\n')

# #Save feature maps to specified directory
# np.save('/scratch/crwlia001/data/densenet201_fmaps/augmented/candidate_layer_1/densenet201_cl1_aug_x_train.npy', x_train)
# np.save('/scratch/crwlia001/data/densenet201_fmaps/augmented/candidate_layer_1/densenet201_cl1_aug_x_val.npy', x_val)
# np.save('/scratch/crwlia001/data/densenet201_fmaps/augmented/candidate_layer_1/densenet201_cl1_aug_x_test.npy', x_test)

# #Save memory usage and execution time dictionaries to specified dictionary
# np.save('/home/crwlia001/model_combination_results/combination_5/densenet201_cl1_aug_feature_extraction_memory_usage.npy', feature_extraction_memory_usage)
# np.save('/home/crwlia001/model_combination_results/combination_5/densenet201_cl1_aug_feature_extraction_time.npy', feature_extraction_time)