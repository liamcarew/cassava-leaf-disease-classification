#function that produces a JSON-style configuration for multi-grained scanning to follow

#def build_gcforestCS():
#
#    #initialise two dictionary objects: 'config' and 'net'
#    config = {}
#    net = {}

#    #set the outputs layers for two different forest types used (ExtraTreesClassifier and RandomForestClassifier)
#    net["outputs"] = []
#    net["outputs"].append("win1/4x4/ets")
#    net["outputs"].append("win1/4x4/rf")

#    #set hyperparameters that will be used when adding a cascade layer during multi-grained scanning
#    layers_sw = {}
#    layers_sw["type"] = "FGWinLayer"
#    layers_sw["name"] = "win1/4x4"
#    layers_sw["bottoms"] = []
#    layers_sw["bottoms"].append("X")
#    layers_sw["bottoms"].append("y")
#    layers_sw["tops"] = []
#    layers_sw["tops"].append("win1/4x4/ets")
#    layers_sw["tops"].append("win1/4x4/rf")
#    layers_sw["n_classes"] = 5
#    layers_sw["estimators"] = []
#    layers_sw["estimators"].append(
#        {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":40,"min_samples_leaf":10})
#    layers_sw["estimators"].append(
#        {"n_folds":5,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":40,"min_samples_leaf":10})
#    layers_sw["stride_x"] = 2
#    layers_sw["stride_y"] = 2
#    layers_sw["win_x"] = 4
#    layers_sw["win_y"] = 4

    #next, we will be performing average pooling on the results
    # layers_pool = {}
    # layers_pool["type"] = "FGPoolLayer"
    # layers_pool["name"] = "pool1"
    # layers_pool["bottoms"] = []
    # layers_pool["bottoms"].append("win1/4x4/ets")
    # layers_pool["bottoms"].append("win1/4x4/rf")
    # layers_pool["tops"] = []
    # layers_pool["tops"].append("pool1/4x4/ets")
    # layers_pool["tops"].append("pool1/4x4/rf")
    # layers_pool["pool_method"] = "avg"
    # layers_pool["win_x"] = 2
    # layers_pool["win_y"] = 2

    #append these hyperparameter settings for multi-grained scanning to the 'net' dictionary
#    layers = []
#    layers.append(layers_sw)
#    #layers.append(layers_pool)
#    net["layers"] = layers

    #create the architecture for cascade forest with confidence screening (CS)
#    ca_config = {}
#    ca_config["random_state"] = 1
#    ca_config["max_layers"] = 20
#    ca_config["early_stopping_rounds"] = 3
#    ca_config["n_classes"] = 5
#    ca_config["estimators"] = []
#    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 50, "max_depth": 10, "n_jobs": 40})
#    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 50, "max_depth": 10, "n_jobs": 40})

    #lastly, append the 'net' dictionary to the 'config' dictionary to finalise MGS structure
#    config["net"] = net
#    config["cascadeCS"] = ca_config

    #return MGS structure in JSON format
#    return config

def build_gcforestCS(n_estimators_mgs, tree_diversity_mgs, n_estimators_ca, tree_diversity_ca):

    config = {}
    net = {}

    net["outputs"] = []
    # net["outputs"].append("win1/4x4/ets")
    # net["outputs"].append("win1/4x4/rf")
    #net["outputs"].append("pool/5x5/ets")
    net["outputs"].append("pool/5x5/rf")
    #net["outputs"].append("pool/7x7/ets")
    net["outputs"].append("pool/7x7/rf")
    #net["outputs"].append("pool/9x9/ets")
    net["outputs"].append("pool/9x9/rf")

    if tree_diversity_mgs:
      net["outputs"].append("pool/5x5/ets")
      net["outputs"].append("pool/7x7/ets")
      net["outputs"].append("pool/9x9/ets")

    #5x5 sliding window
    layer_5x5 = {}
    layer_5x5["type"] = "FGWinLayer"
    layer_5x5["name"] = "win/5x5"
    layer_5x5["bottoms"] = []
    layer_5x5["bottoms"].append("X")
    layer_5x5["bottoms"].append("y")
    layer_5x5["tops"] = []
    #layer_5x5["tops"].append("win/5x5/ets")
    layer_5x5["tops"].append("win/5x5/rf")
    layer_5x5["n_classes"] = 5
    layer_5x5["estimators"] = []
    layer_5x5["estimators"].append(
        {"n_folds":5,"type":"RandomForestClassifier","n_estimators": n_estimators_mgs,"max_depth": 10,"n_jobs":40,"min_samples_leaf":10})
    
    if tree_diversity_mgs:
      layer_5x5["tops"].append("win/5x5/ets")
      layer_5x5["estimators"].append(
          {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators": n_estimators_mgs,"max_depth": 10, "n_jobs":40, "min_samples_leaf":10})

    layer_5x5["stride_x"] = 2
    layer_5x5["stride_y"] = 2
    layer_5x5["win_x"] = 5
    layer_5x5["win_y"] = 5

    #7x7 sliding window
    layer_7x7 = {}
    layer_7x7["type"] = "FGWinLayer"
    layer_7x7["name"] = "win/7x7"
    layer_7x7["bottoms"] = []
    layer_7x7["bottoms"].append("X")
    layer_7x7["bottoms"].append("y")
    layer_7x7["tops"] = []
    #layer_7x7["tops"].append("win/7x7/ets")
    layer_7x7["tops"].append("win/7x7/rf")
    layer_7x7["n_classes"] = 5
    layer_7x7["estimators"] = []
    layer_7x7["estimators"].append(
        {"n_folds":5,"type":"RandomForestClassifier","n_estimators": n_estimators_mgs,"max_depth": 10,"n_jobs":40,"min_samples_leaf":10})
    
    if tree_diversity_mgs:
      layer_7x7["tops"].append("win/7x7/ets")
      layer_7x7["estimators"].append(
          {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators": n_estimators_mgs,"max_depth": 10, "n_jobs":40, "min_samples_leaf":10})

    layer_7x7["stride_x"] = 2
    layer_7x7["stride_y"] = 2
    layer_7x7["win_x"] = 7
    layer_7x7["win_y"] = 7

    #9x9 sliding window
    layer_9x9 = {}
    layer_9x9["type"] = "FGWinLayer"
    layer_9x9["name"] = "win/9x9"
    layer_9x9["bottoms"] = []
    layer_9x9["bottoms"].append("X")
    layer_9x9["bottoms"].append("y")
    layer_9x9["tops"] = []
    #layer_9x9["tops"].append("win/9x9/ets")
    layer_9x9["tops"].append("win/9x9/rf")
    layer_9x9["n_classes"] = 5
    layer_9x9["estimators"] = []
    layer_9x9["estimators"].append(
        {"n_folds":5,"type":"RandomForestClassifier","n_estimators": n_estimators_mgs,"max_depth": 10,"n_jobs":40,"min_samples_leaf":10})
    
    if tree_diversity_mgs:
      layer_9x9["tops"].append("win/9x9/ets")
      layer_9x9["estimators"].append(
          {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators": n_estimators_mgs,"max_depth": 10, "n_jobs":40, "min_samples_leaf":10})

    layer_9x9["stride_x"] = 2
    layer_9x9["stride_y"] = 2
    layer_9x9["win_x"] = 9
    layer_9x9["win_y"] = 9

    #pooling layer
    layers_pool = {}
    layers_pool["type"] = "FGPoolLayer"
    layers_pool["name"] = "pool"
    layers_pool["bottoms"] = []
    #layers_pool["bottoms"].append("win/5x5/ets")
    layers_pool["bottoms"].append("win/5x5/rf")
    #layers_pool["bottoms"].append("win/7x7/ets")
    layers_pool["bottoms"].append("win/7x7/rf")
    #layers_pool["bottoms"].append("win/9x9/ets")
    layers_pool["bottoms"].append("win/9x9/rf")

    if tree_diversity_mgs:
      layers_pool["bottoms"].append("win/5x5/ets")
      layers_pool["bottoms"].append("win/7x7/ets")
      layers_pool["bottoms"].append("win/9x9/ets")

    layers_pool["tops"] = []
    #layers_pool["tops"].append("pool/5x5/ets")
    layers_pool["tops"].append("pool/5x5/rf")
    #layers_pool["tops"].append("pool/7x7/ets")
    layers_pool["tops"].append("pool/7x7/rf")
    #layers_pool["tops"].append("pool/9x9/ets")
    layers_pool["tops"].append("pool/9x9/rf")

    if tree_diversity_mgs:
      layers_pool["tops"].append("pool/5x5/ets")
      layers_pool["tops"].append("pool/7x7/ets")
      layers_pool["tops"].append("pool/9x9/ets")

    layers_pool["pool_method"] = "avg"
    layers_pool["win_x"] = 2
    layers_pool["win_y"] = 2

    layers = []
    layers.append(layer_5x5)
    layers.append(layer_7x7)
    layers.append(layer_9x9)
    layers.append(layers_pool)
    net["layers"] = layers

    #cascade module
    ca_config = {}
    ca_config["random_state"] = 1
    ca_config["max_layers"] = 20
    ca_config["early_stopping_rounds"] = 3
    # ca_config["look_indexs_cycle"] = []
    # ca_config["look_indexs_cycle"].append([0, 1])
    # ca_config["look_indexs_cycle"].append([2, 3])
    # ca_config["look_indexs_cycle"].append([4, 5])
    ca_config["n_classes"] = 5
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": n_estimators_ca, "max_depth": 10, "n_jobs": 40})

    if tree_diversity_ca:
      ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": n_estimators_ca, "max_depth": 10, "n_jobs": 40})

    config["net"] = net
    config["cascadeCS"] = ca_config
    return config