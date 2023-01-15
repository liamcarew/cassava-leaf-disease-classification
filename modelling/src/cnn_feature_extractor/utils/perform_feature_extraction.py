#import necessary libraries

##feature extraction
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow import config
#from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from tensorflow.keras.backend import clear_session
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.get_peak_gpu_mem_usage import get_peak_gpu_mem_usage

#memory and execution time measurement
import time
import tracemalloc

def perform_feature_extraction(x_train, y_train, x_val, y_val, x_test, y_test, cnn_backbone_name, candidate_layer_name, load_fine_tuned_model=False, best_dropout_rate=None, fine_tuned_weights_path=None, use_gpu=False):
    """
    Performs CNN feature extraction based on the specifications in the parameter settings.

    Args:
        x_train (float): NumPy array of training images
        y_train (int): NumPy array of training image labels
        x_val (float): NumPy array of validation images
        y_val (int): NumPy array of validation image labels
        x_test (float): NumPy array of test images
        y_test (int): NumPy array of test image labels
        cnn_backbone_name (str): The name of the transfer learning model to load (either 'DenseNet201' or 'MobileNetV2')
        candidate_layer_name (str): The name of the layer in the convolutional module from which to extract feature maps
        load_fine_tuned_model (bool, optional): Whether to load a weights vector from a pre-trained model before building the feature extractor. Defaults to False.
        best_dropout_rate (int, optional): The dropout rate of the pre-trained models from which weights are being loaded. Defaults to None.
        fine_tuned_weights_path (str, optional): the path to the pre-trained weights vector. Defaults to None.
        use_gpu (bool, optional): Whether a GPU was used to perform the feature extraction. Defaults to False.

    Returns:
        training_data: feature maps extracted from training images
        validation_data: feature maps extracted from validation images
        testing_data: feature maps extracted from test images
        feature_extraction_memory_usage: dictionary of peak RAM usage during CNN feature extraction across the data splits
        feature_extraction_time: dictionary of execution time during CNN feature extraction across the data splits
    """
    #to allow for GPU RAM measurement, need to configure GPU
    if use_gpu:
        gpu_devices = config.list_physical_devices('GPU')

        for gpu in gpu_devices:
            config.experimental.set_memory_growth(gpu, True)

    # ################### Importing Data #############################

    # #training set
    # print('Loading training images and associated labels...\n')

    # if training_set_type == 'original':
    #     x_train = np.load('/scratch/crwlia001/data/training_set/original/x_train.npy')
    #     y_train = np.load('/scratch/crwlia001/data/y_train.npy')

    # elif training_set_type == 'balanced':
    #     x_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_x_train.npy')
    #     y_train = np.load('/scratch/crwlia001/data/training_set/balanced/balanced_y_train.npy')

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

    ################# Data preparation #######################################
    
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

    ################## CNN Feature Extraction ################################

    print('Building CNN feature extractor...\n')

    #Build pre-trained CNN feature extractor using model run specifications
    cnn_feature_extractor = build_feature_extractor(input_image_shape = (224, 224, 3), cnn_backbone_name = cnn_backbone_name, output_layer_name = candidate_layer_name, load_fine_tuned_model=load_fine_tuned_model, best_dropout_rate=best_dropout_rate, fine_tuned_weights_path=fine_tuned_weights_path)

    print('Performing feature extraction...\n')

    ##training set

    #initialise memory usage and execution time measurement variables
    tracemalloc.start()
    start_time_training = time.perf_counter()

    # #initialise variables to monitor RAM usage and execution time during FE
    # peak_ram_before_fe_train = get_peak_gpu_mem_usage(gpu_devices)
    # start_time_fe_train = time.process_time()

    training_data = cnn_feature_extractor.predict(x = training_data, verbose = 1)

    # #terminate monitoring of RAM usage and execution time
    # peak_ram_after_fe_train = get_peak_gpu_mem_usage(gpu_devices)
    # end_time_fe_train = time.process_time()

    #terminate monitoring of RAM usage and execution time
    end_time_training = time.perf_counter()
    first_size, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # #determine training time and RAM usage during training
    # fe_peak_ram_train = round(first_peak / 1000000, 3) #convert from bytes to megabytes
    # fe_exec_time_train = round(end_time_training - start_time_training, 3) #in seconds

    #determine execution time
    training_exec_time = end_time_training - start_time_training

    #save memory usage and execution time to respective dictionaries
    feature_extraction_memory_usage['train'] = first_peak / 1000000 #convert from bytes to megabytes
    feature_extraction_time['train'] = training_exec_time #in seconds

    # #save memory usage and execution time to respective dictionaries
    # feature_extraction_memory_usage['train'] = fe_peak_ram_train #convert from bytes to megabytes
    # feature_extraction_time['train'] = fe_exec_time_train #in seconds

    #clear session - resets RAM measurement
    #clear_session()

    ##validation set

    # #initialise variables to monitor RAM usage and execution time during FE
    # peak_ram_before_fe_val = get_peak_gpu_mem_usage(gpu_devices)
    # start_time_fe_val = time.process_time()

    # validation_data = cnn_feature_extractor.predict(x = validation_data, verbose = 1)

    # #terminate monitoring of RAM usage and execution time
    # peak_ram_after_fe_val = get_peak_gpu_mem_usage(gpu_devices)
    # end_time_fe_val = time.process_time()

    # #determine execution time and RAM usage during validation
    # fe_peak_ram_val = round((peak_ram_after_fe_val - peak_ram_before_fe_val) / 1000000, 3) #convert from bytes to megabytes
    # fe_exec_time_val = round(end_time_fe_val - start_time_fe_val, 3) #in seconds

    #initialise memory usage and execution time measurement variables
    tracemalloc.start()
    start_time_val = time.perf_counter()

    validation_data = cnn_feature_extractor.predict(x = validation_data, verbose = 1)

    #terminate monitoring of RAM usage and execution time
    end_time_val = time.perf_counter()
    second_size, second_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    #determine execution time
    val_exec_time = end_time_val - start_time_val

    #save memory usage and execution time to respective dictionaries
    feature_extraction_memory_usage['val'] = round(second_peak / 1000000, 3) #convert from bytes to megabytes
    feature_extraction_time['val'] = val_exec_time #in seconds

    #clear session - resets RAM measurement
    #clear_session()

    ##test set

    # #initialise variables to monitor RAM usage and execution time during FE
    # peak_ram_before_fe_test = get_peak_gpu_mem_usage(gpu_devices)
    # start_time_fe_test = time.process_time()

    #initialise memory usage and execution time measurement variables
    tracemalloc.start()
    start_time_test = time.perf_counter()

    testing_data = cnn_feature_extractor.predict(x = testing_data, verbose = 1)

    # #terminate monitoring of RAM usage and execution time
    # peak_ram_after_fe_test = get_peak_gpu_mem_usage(gpu_devices)
    # end_time_fe_test = time.process_time()

    #terminate monitoring of RAM usage and execution time
    end_time_test = time.perf_counter()
    third_size, third_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    #determine execution time
    test_exec_time = end_time_test - start_time_test

    # #determine execution time and RAM usage during testing
    # fe_peak_ram_test = round(third_peak / 1000000, 3) #convert from bytes to megabytes
    # fe_exec_time_test = round(test_exec_time, 3) #in seconds

    #save memory usage and execution time to respective dictionaries
    feature_extraction_memory_usage['test'] = round(third_peak / 1000000, 3) #convert from bytes to megabytes
    feature_extraction_time['test'] = round(test_exec_time, 3) #in seconds

    #Convert all to np arrays
    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    testing_data = np.array(testing_data)

    print('Feature extraction complete!\n')

    return training_data, validation_data, testing_data, feature_extraction_memory_usage, feature_extraction_time

    # #before we save Fmaps as well as RAM and time measurement dictionaries, we need variables tracking finetuning to have either 'yes' or 'no' as values
    # if load_fine_tuned_model:
    #     cnn_fine_tuned = 'yes'
    # else:
    #     cnn_fine_tuned = 'no'

    # #Save feature maps to specified directory
    # np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_train.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), training_data)
    # np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_val.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), validation_data)
    # np.save('/scratch/crwlia001/data/{}_fmaps/finetuned_{}/{}/candidate_layer_{}/{}_cl{}_{}_finetuned{}_x_test.npy'.format(cnn_backbone_name, cnn_fine_tuned, training_set_type, candidate_layer_num, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), testing_data)

    # #Save memory usage and execution time dictionaries to specified dictionary
    # np.save('/home/crwlia001/model_combination_results/combination_{}/{}_cl{}_{}_finetuned{}_fe_memory_usage.npy'.format(model_combination, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), feature_extraction_memory_usage)
    # np.save('/home/crwlia001/model_combination_results/combination_{}/{}_cl{}_{}_finetuned{}_fe_execution_time.npy'.format(model_combination, cnn_backbone_name, candidate_layer_num, training_set_type, cnn_fine_tuned), feature_extraction_time)