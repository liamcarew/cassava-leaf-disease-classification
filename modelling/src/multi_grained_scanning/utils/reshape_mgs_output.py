#function that takes the results from MGS, performs necessary reshaping of MGS outputs and finally feeds this vectors into cascade forest classifier

def reshape_mgs_output(mgs_output):
    
    #reshape output vector from MGS to format required for cascade forest classifier
    mgs_output = mgs_output.reshape(mgs_output.shape[0], mgs_output.shape[1]*mgs_output.shape[2]*mgs_output.shape[3])

    return mgs_output