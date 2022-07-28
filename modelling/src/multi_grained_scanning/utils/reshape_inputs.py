import numpy as np

#function that takes the results from MGS and performs necessary reshaping of MGS outputs so that they can be fed into the different classifiers

def reshape_inputs(input_array):
    
    #reshape output vector from MGS to format required for cascade forest classifier
    reshaped_input = input_array.reshape(input_array.shape[0], input_array.shape[1]*input_array.shape[2]*input_array.shape[3])

    return reshaped_input