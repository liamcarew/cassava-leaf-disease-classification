#function that produces a JSON-style configuration for multi-grained scanning to follow

def build_gcforestCS():

    #initialise two dictionary objects: 'config' and 'net'
    config = {}
    net = {}

    #set the outputs layers for two different forest types used (ExtraTreesClassifier and RandomForestClassifier)
    net["outputs"] = []
    net["outputs"].append("pool1/4x4/ets")
    net["outputs"].append("pool1/4x4/rf")

    #set hyperparameters that will be used when adding a cascade layer during multi-grained scanning
    layers_sw = {}
    layers_sw["type"] = "FGWinLayer"
    layers_sw["name"] = "win1/4x4"
    layers_sw["bottoms"] = []
    layers_sw["bottoms"].append("X")
    layers_sw["bottoms"].append("y")
    layers_sw["tops"] = []
    layers_sw["tops"].append("win1/4x4/ets")
    layers_sw["tops"].append("win1/4x4/rf")
    layers_sw["n_classes"] = 5
    layers_sw["estimators"] = []
    layers_sw["estimators"].append(
        {"n_folds":10,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":1,"min_samples_leaf":10})
    layers_sw["estimators"].append(
        {"n_folds":10,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":1,"min_samples_leaf":10})
    layers_sw["stride_x"] = 2
    layers_sw["stride_y"] = 2
    layers_sw["win_x"] = 4
    layers_sw["win_y"] = 4

    #next, we will be performing average pooling on the results
    layers_pool = {}
    layers_pool["type"] = "FGPoolLayer"
    layers_pool["name"] = "pool1"
    layers_pool["bottoms"] = []
    layers_pool["bottoms"].append("win1/4x4/ets")
    layers_pool["bottoms"].append("win1/4x4/rf")
    layers_pool["tops"] = []
    layers_pool["tops"].append("pool1/4x4/ets")
    layers_pool["tops"].append("pool1/4x4/rf")
    layers_pool["pool_method"] = "avg"
    layers_pool["win_x"] = 2
    layers_pool["win_y"] = 2

    #append these hyperparameter settings for multi-grained scanning to the 'net' dictionary
    layers = []
    layers.append(layers_sw)
    layers.append(layers_pool)
    net["layers"] = layers

    #create the architecture for cascade forest with confidence screening (CS)
    ca_config = {}
    ca_config["random_state"] = 1
    ca_config["max_layers"] = 20
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 5
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 10, "type": "RandomForestClassifier", "n_estimators": 50, "max_depth": 10, "n_jobs": 1})
    ca_config["estimators"].append({"n_folds": 10, "type": "ExtraTreesClassifier", "n_estimators": 50, "max_depth": 10, "n_jobs": 1})

    #lastly, append the 'net' dictionary to the 'config' dictionary to finalise MGS structure
    config["net"] = net
    config["cascadeCS"] = ca_config

    #return MGS structure in JSON format
    return config