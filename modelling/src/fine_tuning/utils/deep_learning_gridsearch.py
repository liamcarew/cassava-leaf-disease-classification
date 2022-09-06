# from tensorflow.keras.applications.densenet import DenseNet201
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.keras import models
# from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
# from tensorflow.keras.metrics import SparseCategoricalAccuracy
# from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.keras.losses import categorical_crossentropy
#from tensorflow.data import Dataset, AUTOTUNE
import numpy as np
from tensorflow import config
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed
from tensorflow.keras.backend import clear_session
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#from itertools import product
import time
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.build_deep_learning_model import build_deep_learning_model
from cassava_leaf_disease_classification.modelling.src.fine_tuning.utils.get_peak_gpu_mem_usage import get_peak_gpu_mem_usage

def deep_learning_gridsearch(hyperparameter_combinations, model_combination_num, backbone, training_data, validation_data, x_val, y_val, num_epochs, random_state, start_fine_tune_layer_name, es_patience, gpu_devices):

  assert backbone in ['DenseNet201', 'MobileNetV2'], 'backbone must either be \'DenseNet201\' or \'MobileNetV2\''

  #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the confusion matrix array (value)
  conf_mats = {}
  #conf_mats_val = {}
  #weighted_f1_scores_val = {}
  #oa_val = {}
  #conf_mats_test = {}
  #weighted_f1_scores_test = {}
  #oa_test = {}

  #create an empty dictionary which will be populated with the hyperparameter combination (key) along with the peak RAM usage during training and prediction (value)
  #mem_usage_training = {}
  #mem_usage_prediction_val = {}
  #mem_usage_prediction_test = {}

  #create empty dictionaries which will be populated with the hyperparameter combination (key) along with the execution times for training and prediction (value)
  #training_time = {}
  #prediction_time_val = {}
  #prediction_time_test = {}

  #Finally, create a dictionary to measure overall execution time during the gridsearch
  #hyp_gridsearch_time = {}
  hyp_comb_results = {}

  #start measurement of hyperparameter gridsearch execution time
  start_time_hyp_gridsearch = time.process_time()

  for comb in hyperparameter_combinations:

    current_comb_results = {}

    #save each of the hyperparameters in this combination as separate variables
    dropout_rate, optimiser, learning_rate = comb

    #create deep learning model instance with these hyperparameters
    model = build_deep_learning_model(backbone = backbone,
                                      dropout_rate = dropout_rate,
                                      optimiser = optimiser,
                                      learning_rate = learning_rate,
                                      start_fine_tune_layer_name = start_fine_tune_layer_name)

    #add early stopping as callback to model to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss',
                                  patience=es_patience,
                                  verbose=0,
                                  restore_best_weights=True)

    #specify number of epochs and batch size
    EPOCHS = num_epochs
    #BATCH_SIZE = batch_size        

    #initialise variables to monitor RAM usage and execution time during training
    peak_ram_before_model_training = get_peak_gpu_mem_usage(gpu_devices)
    start_time_training = time.process_time()

    #fit model to training data
    set_seed(random_state) #for reproducibility
    model.fit(training_data,
              #batch_size = BATCH_SIZE,
              epochs = EPOCHS,
              validation_data = validation_data,
              callbacks = [early_stopping],
              shuffle=False,
              verbose=0
              )

    #terminate monitoring of RAM usage and execution time
    peak_ram_after_model_training = get_peak_gpu_mem_usage(gpu_devices)
    end_time_training = time.process_time()

    #determine training time and RAM usage during training
    training_peak_ram = round((peak_ram_after_model_training - peak_ram_before_model_training) / 1000000, 3) #convert from bytes to megabytes
    training_exec_time = round(end_time_training - start_time_training, 3) #in seconds

    #assign peak RAM usage and execution time to respective dictionaries
    current_comb_results['training_peak_mem_usage'] = training_peak_ram
    current_comb_results['training_runtime'] = training_exec_time
    #mem_usage_training[str(comb)] = training_peak_ram
    #training_time[str(comb)] = training_exec_time

    #save weights vector
    hyp_settings = [str(dropout_rate), optimiser, str(learning_rate)]
    model.save_weights('/scratch/crwlia001/fine_tuned_model_weights/{}/model_comb_{}_{}.h5'.format(backbone, model_combination_num, '_'.join(hyp_settings)))
    #model.save_weights('/content/drive/MyDrive/model_weights/{}/model_comb_{}_{}.h5'.format(backbone, model_combination_num, '_'.join(hyp_settings)))

    ##perform predictions on validation set

    #initialise measurement variables
    peak_ram_before_val_predictions = get_peak_gpu_mem_usage(gpu_devices)
    start_time_predictions_val = time.process_time()

    #perform validation set predictions
    y_val_pred = np.argmax(model.predict(x_val), axis=1)

    #terminate measurement variables
    peak_ram_after_val_predictions = get_peak_gpu_mem_usage(gpu_devices)
    end_time_predictions_val = time.process_time()

    #determine peak RAM usage and prediction time
    peak_ram_during_val_predictions = round((peak_ram_after_val_predictions - peak_ram_before_val_predictions) / 1000000, 3)#convert from bytes to megabytes
    prediction_exec_time_val = round(end_time_predictions_val - start_time_predictions_val, 3) #in seconds

    #assign peak memory usage and prediction times to respective dictionaries
    current_comb_results['val_pred_peak_mem_usage'] = peak_ram_during_val_predictions
    current_comb_results['val_pred_runtime'] = prediction_exec_time_val
    #mem_usage_prediction_val[str(comb)] = peak_ram_during_val_predictions
    #prediction_time_val[str(comb)] = prediction_exec_time_val

    #produce confusion matrix
    cf_matrix_val = confusion_matrix(y_val, y_val_pred)
    conf_mats[str(comb)] = cf_matrix_val

    #calculate weighted f1-score and overall accuracy and add these to their respective dictionaries
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    oa = accuracy_score(y_val, y_val_pred)

    current_comb_results['val_weighted_f1_score'] = round(f1, 4)
    current_comb_results['val_overall_acc'] = round(oa, 4)
    #weighted_f1_scores_val[str(comb)] = round(f1, 2)
    #oa_val[str(comb)] = round(oa, 2)

    #check memory usage before clear_session() call
    # if gpu_devices:
    #   print(tf.config.experimental.get_memory_info('GPU:0'))

    #clear the session before the next model run - allows for accurate peak RAM usage during next run and so OOM error doesn't occur
    config.experimental.reset_memory_stats('GPU:0')
    clear_session()

    #check memory usage after clear_session() call
    # if gpu_devices:
    #   print(tf.config.experimental.get_memory_info('GPU:0'))

    #add 'current_comb_results' to 'hyp_comb_results'
    hyp_comb_results[str(comb)] = current_comb_results

  #end measurement of running time for gridsearch
  end_time_hyp_gridsearch = time.process_time()

  #determine execution time for gridsearch and add it to dictionary
  hyp_gridsearch_exec_time = end_time_hyp_gridsearch - start_time_hyp_gridsearch
  hyp_comb_results['hyp_gridsearch_time'] = round(hyp_gridsearch_exec_time, 3)

  # #create a nested dictionary of the gridsearch results
  # hyp_gridsearch_results = {}
  # hyp_gridsearch_results['conf_mats_val'], hyp_gridsearch_results['weighted_f1_scores_val'], hyp_gridsearch_results['oa_val'] = conf_mats_val, weighted_f1_scores_val, oa_val
  # hyp_gridsearch_results['mem_usage_training'], hyp_gridsearch_results['mem_usage_prediction_val'] = mem_usage_training, mem_usage_prediction_val
  # hyp_gridsearch_results['training_time'], hyp_gridsearch_results['prediction_time_val'] = training_time, prediction_time_val
  # hyp_gridsearch_results['hyp_gridsearch_time'] = hyp_gridsearch_time

  ################# Saving results from gridsearch ############################

  #hyperparameter gridsearch results
  np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_hyp_comb_results.npy'.format(model_combination_num, model_combination_num), hyp_comb_results)

  #confusion matrices
  np.save('/home/crwlia001/model_combination_results/combination_{}/model_comb_{}_conf_mats.npy'.format(model_combination_num, model_combination_num), conf_mats)
  