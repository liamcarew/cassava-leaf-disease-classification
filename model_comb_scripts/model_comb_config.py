#import necessary libraries

import argparse
import numpy as np
import pickle
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#from sklearn.model_selection import GridSearchCV
import random
from sklearn import utils
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.build_gcForestCS import build_gcforestCS
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.reshape_inputs import reshape_inputs
from cassava_leaf_disease_classification.modelling.src.multi_grained_scanning.utils.gcForestCS.lib.gcforest.gcforestCS import GCForestCS
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.build_feature_extractor import build_feature_extractor
from cassava_leaf_disease_classification.modelling.src.cnn_feature_extractor.utils.perform_feature_extraction import perform_feature_extraction
from itertools import product
import time
import tracemalloc

def gcForestCS_gridsearch(data_paths, hyp_settings, model_combination_num, cnn_feature_extraction=False, feature_extraction_settings=None):

    ###################### Importing Data ###################################

    print('Loading training images/Fmaps and labels...\n')

    x_train = np.load(data_paths['training_images'])
    y_train = np.load(data_paths['training_labels'])

    print('Training images/Fmaps and labels loaded!\n')

    print('Loading validation images/Fmaps and labels...\n')

    x_val = np.load(data_paths['validation_images'])
    y_val = np.load(data_paths['validation_labels'])

    print('Validation images/Fmaps and labels loaded!\n')

    print('Loading test images/Fmaps and labels...\n')

    x_test = np.load(data_paths['test_images'])
    y_test = np.load(data_paths['test_labels'])

    print('Test images/Fmaps and labels loaded!\n')

    ################## CNN Feature Extraction ################################################

    #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the peak RAM usage during training and prediction (value)
    peak_mem_usage = {}
    # mem_usage_training = {}
    # mem_usage_prediction_val = {}
    # mem_usage_prediction_test = {}

    #create empty dictionaries which will be populated with the hyperparameter combination (key) along with the execution times for training and prediction (value)
    execution_time = {}
    # training_time = {}
    # prediction_time_val = {}
    # prediction_time_test = {}

    if cnn_feature_extraction:

        #Perform feature extraction
        x_train, x_val, x_test, feature_extraction_memory_usage, feature_extraction_time = perform_feature_extraction(
            x_train = x_train,
            y_train = y_train,
            x_val = x_val,
            y_val = y_val,
            x_test = x_test,
            y_test = y_test,
            cnn_backbone_name = feature_extraction_settings['cnn_backbone_name'],
            candidate_layer_name = feature_extraction_settings['candidate_layer_name'],
            load_fine_tuned_model = feature_extraction_settings['load_fine_tuned_model'],
            fine_tuned_weights_path = feature_extraction_settings['fine_tuned_weights_path']
            )

        #add measurement results from feature extraction to relevant dictionaries
        peak_mem_usage['feature_extraction'] = feature_extraction_memory_usage
        execution_time['feature_extraction'] = feature_extraction_time

    ################## Performing gcForestCS hyperparameter gridsearch #######################

    print('Performing hyperparameter gridsearch...\n')

    #produce a list of all of the different hyperparameter combinations
    hyperparameter_comb = [_ for _ in product(hyp_settings['combs_mgs'], hyp_settings['combs_pooling_mgs'], hyp_settings['combs_ca'])]

    # Reshape the training and validation inputs to format needed for multi-grained scanning (n_images, n_channels, width, height)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[3], x_val.shape[1], x_val.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[3], x_test.shape[1], x_test.shape[2])

    #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the confusion matrix array (value)
    conf_mats = {}
    #weighted_f1_scores = {}
    #overall_acc = {}
    
    # conf_mats_val = {}
    # weighted_f1_scores_val = {}
    # oa_val = {}
    # conf_mats_test = {}
    # weighted_f1_scores_test = {}
    # oa_test = {}

    # #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the peak RAM usage during training and prediction (value)
    # mem_usage_training = {}
    # mem_usage_prediction_val = {}
    # mem_usage_prediction_test = {}

    # #create empty dictionaries which will be populated with the hyperparameter combination (key) along with the execution times for training and prediction (value)
    # training_time = {}
    # prediction_time_val = {}
    # prediction_time_test = {}

    #Finally, create a dictionary to measure overall execution time during the gridsearch
    #hyp_gridsearch_time = {}
    hyp_comb_results = {}

    #start measurement of hyperparameter gridsearch execution time
    start_time_hyp_gridsearch = time.process_time()

    for comb in hyperparameter_comb:

        current_comb_results = {}
        current_comb_cfmats = {}

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
        current_comb_results['training_peak_mem_usage'] = first_peak / 1000000 #convert from bytes to megabytes
        current_comb_results['training_runtime'] = training_exec_time #in seconds
        #peak_mem_usage['training', str(comb)] = first_peak / 1000000 #convert from bytes to megabytes
        #execution_time['training', str(comb)] = training_exec_time #in seconds

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
        current_comb_results['val_preds_peak_mem_usage'] = second_peak / 1000000 #convert from bytes to megabytes
        current_comb_results['val_preds_runtime'] = prediction_exec_time_val #in seconds
        #peak_mem_usage['val_predictions', str(comb)] = second_peak / 1000000 #convert from bytes to megabytes
        #execution_time['val_predictions', str(comb)] = prediction_exec_time_val #in seconds

        #produce confusion matrix
        cf_matrix_val = confusion_matrix(y_val, y_val_pred)
        current_comb_cfmats['val'] = cf_matrix_val

        #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
        current_comb_results['weighted_f1_val'] = round(f1_score(y_val, y_val_pred, average='weighted'), 6)
        #weighted_f1_scores['val', str(comb)] = round(f1_score(y_val, y_val_pred, average='weighted'), 3)

        #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
        current_comb_results['overall_acc_val'] = round(accuracy_score(y_val, y_val_pred), 6)
        #overall_acc['val', str(comb)] = round(accuracy_score(y_val, y_val_pred), 3)

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
        current_comb_results['test_preds_peak_mem_usage'] = third_peak / 1000000 #convert from bytes to megabytes
        current_comb_results['test_preds_runtime'] = prediction_exec_time_test #in seconds
        #peak_mem_usage['test_predictions', str(comb)] = third_peak / 1000000 #convert from bytes to megabytes
        #execution_time['test_predictions', str(comb)] = prediction_exec_time_test #in seconds

        #produce confusion matrix
        cf_matrix_test = confusion_matrix(y_test, y_test_pred)
        current_comb_cfmats['test'] = cf_matrix_test

        #produce f1-score (this was added in after your model run in case you are wondering why dictionaries are empty)
        current_comb_results['weighted_f1_test'] = round(f1_score(y_test, y_test_pred, average='weighted'), 6)
        #weighted_f1_scores['test', str(comb)] = round(f1_score(y_test, y_test_pred, average='weighted'), 3)

        #produce overall accuracy (this was added in after your model run in case you are wondering why there isn't a dictionary for it)
        current_comb_results['overall_acc_test'] = round(accuracy_score(y_test, y_test_pred), 6)
        #overall_acc['test', str(comb)] = round(accuracy_score(y_test, y_test_pred), 3)

        #add 'current_comb_results' to 'hyp_comb_results'
        hyp_comb_results[str(comb)] = current_comb_results

        #add 'current_comb_cfmats' to 'conf_mats'
        conf_mats[str(comb)] = current_comb_cfmats

    #end measurement of running time for gridsearch
    end_time_hyp_gridsearch = time.process_time()

    #determine execution time for gridsearch and add it to dictionary
    hyp_gridsearch_exec_time = end_time_hyp_gridsearch - start_time_hyp_gridsearch
    hyp_comb_results['hyp_gridsearch_time'] = hyp_gridsearch_exec_time
    #hyp_gridsearch_time['model_comb_{}'.format(model_combination_num)] = hyp_gridsearch_exec_time

    ################# Saving results from gridsearch ############################

    #hyperparameter gridsearch results
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_hyp_comb_results.npy'.format(model_combination_num, model_combination_num), hyp_comb_results)

    #confusion matrices
    np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_conf_mats.npy'.format(model_combination_num, model_combination_num), conf_mats)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_conf_mats_test.npy'.format(model_combination_num, model_combination_num), conf_mats_test)

    #weighted f1-scores
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_weighted_f1_scores.npy'.format(model_combination_num, model_combination_num), weighted_f1_scores)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_f1_test.npy'.format(model_combination_num, model_combination_num), weighted_f1_scores_test)

    #overall accuracy
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_overall_acc.npy'.format(model_combination_num, model_combination_num), overall_acc)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_oa_test.npy'.format(model_combination_num, model_combination_num), oa_test)

    #peak memory usage
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_peak_mem_usage.npy'.format(model_combination_num, model_combination_num), peak_mem_usage)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_mem_usage_prediction_val.npy'.format(model_combination_num, model_combination_num), mem_usage_prediction_val)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_mem_usage_prediction_test.npy'.format(model_combination_num, model_combination_num), mem_usage_prediction_test)

    #training and prediction times
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_execution_times.npy'.format(model_combination_num, model_combination_num), execution_time)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_prediction_time_val.npy'.format(model_combination_num, model_combination_num), prediction_time_val)
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_prediction_time_test.npy'.format(model_combination_num, model_combination_num), prediction_time_test)

    #hyperparameter gridsearch execution time
    #np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_hyp_gridsearch_time.npy'.format(model_combination_num, model_combination_num), hyp_gridsearch_time)