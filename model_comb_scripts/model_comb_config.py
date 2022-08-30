#import necessary libraries

import argparse
import numpy as np
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import random
from sklearn import utils
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.build_gcForestCS import build_gcforestCS
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.reshape_inputs import reshape_inputs
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest.gcforestCS import GCForestCS
from itertools import product
import time
import tracemalloc

def gcForestCS_model_config(x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, y_test_path, combs_mgs, combs_pooling_mgs, combs_ca, model_combination_num):

    ###################### Importing Data ###################################

    print('Loading training feature maps...\n')

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    print('Training feature maps and associated labels loaded!\n')

    print('Loading validation feature maps and associated labels...\n')

    x_val = np.load(x_val_path)
    y_val = np.load(y_val_path)

    print('Validation feature maps and associated labels loaded!\n')

    print('Loading test feature maps and associated labels...\n')

    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    print('Test feature maps and associated labels loaded!\n')

    ################## Performing gcForestCS hyperparameter gridsearch #######################

    print('Performing hyperparameter gridsearch...\n')

    #produce a list of all of the different hyperparameter combinations
    hyperparameter_comb = [_ for _ in product(combs_mgs, combs_pooling_mgs, combs_ca)]

    # Reshape the training and validation inputs to format needed for multi-grained scanning (n_images, n_channels, width, height)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[3], x_val.shape[1], x_val.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2])

    #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the confusion matrix array (value)
    conf_mats_val = {}
    weighted_f1_scores_val = {}
    oa_val = {}
    conf_mats_test = {}
    weighted_f1_scores_test = {}
    oa_test = {}

    #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the peak RAM usage during training and prediction (value)
    mem_usage_training = {}
    mem_usage_prediction_val = {}
    mem_usage_prediction_test = {}

    #create empty dictionaries which will be populated with the hyperparameter combination (key) along with the execution times for training and prediction (value)
    training_time = {}
    prediction_time_val = {}
    prediction_time_test = {}

    #Finally, create a dictionary to measure overall execution time during the gridsearch
    hyp_gridsearch_time = {}

    #start measurement of hyperparameter gridsearch execution time
    start_time_hyp_gridsearch = time.process_time()

    for comb in hyperparameter_comb:

        #assign hyperparameters to variables
        n_estimators_mgs, tree_diversity_mgs = comb[0]
        pooling_mgs = comb[1]
        n_estimators_ca, tree_diversity_ca = comb[2]

        print('Fitting gcForestCS model using the following hyperparameter settings:\nn_estimators: {}, tree_diversity_mgs: {}, n_estimators_ca: {}, tree_diversity_ca: {}\n'.format(n_estimators_mgs, tree_diversity_mgs, n_estimators_ca, tree_diversity_ca))

        #get model configuration
        config = build_gcforestCS(n_estimators_mgs = n_estimators_mgs,
                                    tree_diversity_mgs = tree_diversity_mgs,
                                    pooling_mgs = pooling_mgs,
                                    n_estimators_ca = n_estimators_ca,
                                    tree_diversity_ca = tree_diversity_ca)

        #create a model instance using model configuration
        cnn_gc = GCForestCS(config)

        #initialise variables to monitor RAM usage and execution time during training
        tracemalloc.start()
        #tracemalloc.reset_peak()
        start_time_training = time.process_time()

        #fit model to training data
        cnn_gc.fit_transform(x_train, y_train)

        #terminate monitoring of RAM usage and execution time
        end_time_training = time.process_time()
        first_size, first_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        #determine training time and RAM usage
        training_exec_time = end_time_training - start_time_training
        #memory_usage_training = tracemalloc.get_traced_memory()

        #assign these values to respective dictionaries
        mem_usage_training[str(comb)] = first_peak / 1000000 #convert from bytes to megabytes
        training_time[str(comb)] = training_exec_time #in seconds

        ##repeat the above for predictions
        start_time_predictions_val = time.process_time()
        tracemalloc.start()

        #perform predictions
        y_val_pred = cnn_gc.predict(x_val)

        end_time_predictions_val = time.process_time()
        second_size, second_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        prediction_exec_time_val = end_time_predictions_val - start_time_predictions_val
        #memory_usage_prediction = tracemalloc.get_traced_memory()

        #assign these values to respective dictionaries
        mem_usage_prediction_val[str(comb)] = second_peak / 1000000 #convert from bytes to megabytes
        prediction_time_val[str(comb)] = prediction_exec_time_val #in seconds

        #produce confusion matrix
        cf_matrix_val = confusion_matrix(y_val, y_val_pred)
        conf_mats_val[str(comb)] = cf_matrix_val

        #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
        weighted_f1_scores_val[str(comb)] = round(f1_score(y_val, y_val_pred, average='weighted'), 3)

        #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
        oa_val[str(comb)] = round(accuracy_score(y_val, y_val_pred), 3)

        ##Repeat the above for test set
        start_time_predictions_test = time.process_time()
        tracemalloc.start()

        #perform predictions
        y_test_pred = cnn_gc.predict(x_test)

        end_time_predictions_test = time.process_time()
        third_size, third_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        prediction_exec_time_test = end_time_predictions_test - start_time_predictions_test
        #memory_usage_prediction = tracemalloc.get_traced_memory()

        #assign these values to respective dictionaries
        mem_usage_prediction_test[str(comb)] = third_peak / 1000000 #convert from bytes to megabytes
        prediction_time_test[str(comb)] = prediction_exec_time_test #in seconds

        #produce confusion matrix
        cf_matrix_test = confusion_matrix(y_test, y_test_pred)
        conf_mats_test[str(comb)] = cf_matrix_test

        #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
        weighted_f1_scores_test[str(comb)] = round(f1_score(y_test, y_test_pred, average='weighted'), 3)

        #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
        oa_test[str(comb)] = round(accuracy_score(y_test, y_test_pred), 3)

    #end measurement of running time for gridsearch
    end_time_hyp_gridsearch = time.process_time()

    #determine execution time for gridsearch and add it to dictionary
    hyp_gridsearch_exec_time = end_time_hyp_gridsearch - start_time_hyp_gridsearch
    hyp_gridsearch_time['model_comb_{}'.format(model_combination_num)] = hyp_gridsearch_exec_time

    ################# Saving results from gridsearch ############################

    #confusion matrices
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_conf_mats_val.npy'.format(model_combination_num, model_combination_num), conf_mats_val)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_conf_mats_test.npy'.format(model_combination_num, model_combination_num), conf_mats_test)

    #weighted f1-scores
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_f1_val.npy'.format(model_combination_num, model_combination_num), weighted_f1_scores_val)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_f1_test.npy'.format(model_combination_num, model_combination_num), weighted_f1_scores_test)

    #overall accuracy
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_oa_val.npy'.format(model_combination_num, model_combination_num), oa_val)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_oa_test.npy'.format(model_combination_num, model_combination_num), oa_test)

    #peak memory usage
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_mem_usage_training.npy'.format(model_combination_num, model_combination_num), mem_usage_training)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_mem_usage_prediction_val.npy'.format(model_combination_num, model_combination_num), mem_usage_prediction_val)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_mem_usage_prediction_test.npy'.format(model_combination_num, model_combination_num), mem_usage_prediction_test)

    #training and prediction times
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_training_time.npy'.format(model_combination_num, model_combination_num), training_time)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_prediction_time_val.npy'.format(model_combination_num, model_combination_num), prediction_time_val)
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_prediction_time_test.npy'.format(model_combination_num, model_combination_num), prediction_time_test)

    #hyperparameter gridsearch execution time
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_hyp_gridsearch_time.npy'.format(model_combination_num, model_combination_num), hyp_gridsearch_time)