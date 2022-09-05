#import necessary libraries

##feature extraction
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow import config
#from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor

#memory and execution time measurement
import time
import tracemalloc

def perform_feature_extraction(training_set_type, cnn_backbone_name, candidate_layer_name, candidate_layer_num, model_combination, use_gpu=False, load_fine_tuned_model = False, fine_tuned_weights_path=None):

    #to allow for GPU RAM measurement, need to configure GPU
    if use_gpu:
        gpu_devices = config.list_physical_devices('GPU')

        for gpu in gpu_devices:
            config.experimental.set_memory_growth(gpu, True)

    ################### Importing Data #############################

    #training set
    print('Loading training images and associated labels...\n')

    if training_set_type == 'original':
        x_train = np.load('/scratch/crwlia001/data/training_set/original/x_train.npy')
        y_train = np.load('/scratch/crwlia001/data/y_train.npy')

    elif training_set_type == 'balanced':
        x_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy')
        y_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy')

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
    cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = cnn_backbone_name, output_layer_name = candidate_layer_name, load_fine_tuned_model=load_fine_tuned_model, fine_tuned_weights_path=fine_tuned_weights_path)

    print('Performing feature extraction...\n')

    #Extract feature maps from pre-trained CNN using 'imagenet' weights
    BATCH_SIZE = 32

    feature_extraction_memory_usage = {}
    feature_extraction_time = {}

    #prepare data for feature extraction
    training_data = Dataset.from_tensor_slices((x_train, y_train)).cache().batch(batch_size = BATCH_SIZE)
    validation_data = Dataset.from_tensor_slices((x_val, y_val)).cache().batch(batch_size = BATCH_SIZE)
    testing_data = Dataset.from_tensor_slices((x_test, y_test)).cache().batch(batch_size = BATCH_SIZE)

    #delete 'x_train', 'y_train', 'x_val', 'y_val', 'x_test' and 'y_test' to free up memory
    del x_train, y_train, x_val, y_val, x_test, y_test

    ##training set

    #initialise memory usage and execution time measurement variables
    #tracemalloc.start()
    start_time_training = time.process_time()

    training_data = cnn_feature_extractor.predict(x = training_data, verbose = 1)

    #terminate monitoring of RAM usage and execution time
    end_time_training = time.process_time()
    #first_size, first_peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #determine execution time
    training_exec_time = end_time_training - start_time_training

    #save memory usage and execution time to respective dictionaries
    #feature_extraction_memory_usage['train'] = first_peak / 1000000 #convert from bytes to megabytes
    feature_extraction_time['train'] = training_exec_time #in seconds

    ##validation set

    #initialise memory usage and execution time measurement variables
    #tracemalloc.start()
    start_time_val = time.process_time()

    validation_data = cnn_feature_extractor.predict(x = validation_data, verbose = 1)

    #terminate monitoring of RAM usage and execution time
    end_time_val = time.process_time()
    #second_size, second_peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #determine execution time
    val_exec_time = end_time_val - start_time_val

    #save memory usage and execution time to respective dictionaries
    #feature_extraction_memory_usage['val'] = second_peak / 1000000 #convert from bytes to megabytes
    feature_extraction_time['val'] = val_exec_time #in seconds

    ##test set

    #initialise memory usage and execution time measurement variables
    #tracemalloc.start()
    start_time_test = time.process_time()

    testing_data = cnn_feature_extractor.predict(x = testing_data, verbose = 1)

    #terminate monitoring of RAM usage and execution time
    end_time_test = time.process_time()
    #third_size, third_peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #determine execution time
    test_exec_time = end_time_test - start_time_test

    #save memory usage and execution time to respective dictionaries
    #feature_extraction_memory_usage['test'] = third_peak / 1000000 #convert from bytes to megabytes
    feature_extraction_time['test'] = test_exec_time #in seconds

    #Convert all to np arrays
    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    testing_data = np.array(testing_data)

    print('Feature extraction complete!\n')

    #before we save Fmaps as well as RAM and time measurement dictionaries, we need variables tracking finetuning to have either 'yes' or 'no' as values
    if load_fine_tuned_model:
        cnn_fine_tuned = 'yes'
    else:
        cnn_fine_tuned = 'no'

    #Save feature maps to specified directory
    np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_train.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), training_data)
    np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_val.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), validation_data)
    np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_test.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), testing_data)

    #Save memory usage and execution time dictionaries to specified dictionary
    np.save('/home/crwlia001/model_combination_results/combination_{}/{}_cl{}_{}_finetuned{}_fe_memory_usage.npy'.format(model_combination, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), feature_extraction_memory_usage)
    np.save('/home/crwlia001/model_combination_results/combination_{}/{}_cl{}_{}_finetuned{}_fe_execution_time.npy'.format(model_combination, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), feature_extraction_time)