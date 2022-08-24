#import necessary libraries

##feature extraction
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor

#memory and execution time measurement
import time
import tracemalloc

################### Importing Data #############################

#training set
print('Loading training images and associated labels...\n')

x_train = np.load('/scratch/crwlia001/data/original_training_set/x_train.npy')
y_train = np.load('/scratch/crwlia001/data/y_train.npy')

print('training images and associated labels loaded!\n')

#validation set
print('Loading validation images and associated labels...\n')

x_val = np.load('/scratch/crwlia001/data/x_val.npy')
y_val = np.load('/scratch/crwlia001/data/y_val.npy')

print('validation images and associated labels loaded!\n')

#test set
print('Loading test images and associated labels...\n')

x_test = np.load('/scratch/crwlia001/data/x_test.npy')
y_test = np.load('/scratch/crwlia001/data/y_test.npy')

print('test images and associated labels loaded!\n')

################## CNN Feature Extraction ################################

print('Building CNN feature extractor...\n')

#Build pre-trained CNN feature extractor using model run specifications
cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = 'DenseNet201', output_layer_name = 'pool3_conv') #(28x28x256)

print('Performing feature extraction...\n')

#Extract feature maps from pre-trained CNN using 'imagenet' weights
BATCH_SIZE = 32

feature_extraction_memory_usage = {}
feature_extraction_time = {}

##training set

#initialise memory usage and execution time measurement variables
tracemalloc.start()
start_time_training = time.process_time()

x_train = cnn_feature_extractor.predict(x = x_train,
                                        batch_size = BATCH_SIZE,
                                        verbose = 1)

#terminate monitoring of RAM usage and execution time
end_time_training = time.process_time()
first_size, first_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

#determine execution time
training_exec_time = end_time_training - start_time_training

#save memory usage and execution time to respective dictionaries
feature_extraction_memory_usage['train'] = first_peak / 1000000 #convert from bytes to megabytes
feature_extraction_time['train'] = training_exec_time #in seconds

##validation set

#initialise memory usage and execution time measurement variables
tracemalloc.start()
start_time_val = time.process_time()

x_val = cnn_feature_extractor.predict(x = x_val, batch_size = BATCH_SIZE, verbose = 1)

#terminate monitoring of RAM usage and execution time
end_time_val = time.process_time()
second_size, second_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

#determine execution time
val_exec_time = end_time_val - start_time_val

#save memory usage and execution time to respective dictionaries
feature_extraction_memory_usage['val'] = second_peak / 1000000 #convert from bytes to megabytes
feature_extraction_time['val'] = val_exec_time #in seconds

##test set

#initialise memory usage and execution time measurement variables
tracemalloc.start()
start_time_test = time.process_time()

x_test = cnn_feature_extractor.predict(x = x_test, batch_size = BATCH_SIZE, verbose = 1)

#terminate monitoring of RAM usage and execution time
end_time_test = time.process_time()
third_size, third_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

#determine execution time
test_exec_time = end_time_test - start_time_test

#save memory usage and execution time to respective dictionaries
feature_extraction_memory_usage['test'] = third_peak / 1000000 #convert from bytes to megabytes
feature_extraction_time['test'] = test_exec_time #in seconds

#Convert all to np arrays
x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

print('Feature extraction complete!\n')

#Save feature maps to specified directory
np.save('/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_train.npy', x_train)
np.save('/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_val.npy', x_val)
np.save('/scratch/crwlia001/data/densenet201_fmaps/original/candidate_layer_1/densenet201_cl1_og_test.npy', x_test)

#Save memory usage and execution time dictionaries to specified dictionary
np.save('/home/crwlia001/combination_3/densenet201_feature_extraction_memory_usage_og.npy', feature_extraction_memory_usage)
np.save('/home/crwlia001/combination_3/densenet201_feature_extraction_time_og.npy', feature_extraction_time)