# Curated training set?: no curation (nc)
# Augmentation?: no augmentation (na)
# CNN backbone: DenseNet201 (Backbone 1)(b1)
# Candidate layer: 28x28x128 (cl1)

import argparse
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
#import tensorflow as tf
import tensorflow_datasets as tfds
#from deepforest import CascadeForestClassifier
import random
from sklearn import utils
from image_pre_processing.utils.data_generators_with_no_aug import image_preprocessing
from image_pre_processing.utils.convert_to_np_array import convert_to_np_array
#from modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest import gcforestCS
from modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

###################### Importing Data ###################################

print('Importing data...\n')

##read in cassava dataset
dataset = tfds.load('cassava')

print('Data import complete!\n')

##################### Image Pre-processing #############################

print('Performing image pre-processing...\n')

##apply image pre-processing to each data split
training_data = dataset['train'].map(image_preprocessing)
validation_data = dataset['validation'].map(image_preprocessing)
#test_data = dataset['test'].map(image_preprocessing)

##produce np arrays of the training set and validation set class labels respectively
y_train = convert_to_np_array(dataset = training_data, split_name = 'training')
y_val = convert_to_np_array(dataset = validation_data, split_name = 'validation')

##prepare training and validation image sets for processing in feature extractor

#define global variables
#AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16

#prepare training and validation image sets
x_train = training_data.cache().batch(batch_size)
x_val = validation_data.cache().batch(batch_size)

print('Image Pre-processing complete!\n')

################## CNN Feature Extraction ################################

print('Building CNN feature extractor...\n')

##Build pre-trained CNN feature extractor using model run specifications
cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = 'DenseNet201', output_layer_name = 'conv3_block12_2_conv') #28x28x32 (as example case and then change to 28x28x128 once you sort out things your side)

print('Performing feature extraction...\n')

##Fit training data and validation data to feature extractor to produce feature maps using 'imagenet' weights
x_train = cnn_feature_extractor.predict(x = x_train,
                                        batch_size = batch_size,
                                        verbose = 1)
x_val = cnn_feature_extractor.predict(x = x_val, batch_size = batch_size, verbose = 1)

##Convert both to np arrays
x_train = np.array(x_train)
x_val = np.array(x_val)

print('Feature extraction complete!\n')

print('Saving training feature maps...\n')

np.save('x_train_feature_maps.npy', x_train)

print('Training feature maps saved!\n')

print('Saving validation feature maps...\n')

np.save('x_val_feature_maps.npy', x_val)

print('Validation feature maps saved!\n')

