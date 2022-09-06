import numpy as np
from tensorflow.data import Dataset, AUTOTUNE

def deep_learning_data_preparation(data_paths, batch_size):

    #read in the images and labels for the training and validation sets
    print('Loading training images/Fmaps and labels...\n')

    x_train = np.load(data_paths['training_images'])
    y_train = np.load(data_paths['training_labels'])

    print('Training images/Fmaps and labels loaded!\n')

    print('Loading validation images/Fmaps and labels...\n')

    x_val = np.load(data_paths['validation_images'])
    y_val = np.load(data_paths['validation_labels'])

    print('Validation images/Fmaps and labels loaded!\n')

    #define some constants before preparing data
    BATCH_SIZE = batch_size

    #prepare data for deep learning
    training_data = Dataset.from_tensor_slices((x_train, y_train)).cache().batch(batch_size = BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_data = Dataset.from_tensor_slices((x_val, y_val)).cache().batch(batch_size = BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    #delete unnecessary variables 'x_train' and 'y_train' but keep 'x_val' and 'y_val' as these will be needed for predictions later on
    del x_train
    del y_train

    return training_data, validation_data, x_val, y_val
