"""
IMDB datasets demo for gcforestCS
Usage:
    define the model within scripts:
        python main_gcforestCS.py
    get config from json file:
        python main_gcforestCS.py --model imdb-gcForestCS.json

Description: A python 2.7 implementation of gcForestCS proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] M. Pang, K. M. Ting, P. Zhao, and Z.-H. Zhou. Improving deep forest by confidence screening. In ICDM-2018.  (http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm18.pdf)
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package is developed by Mr. Ming Pang(pangm@lamda.nju.edu.cn), which is based on the gcForest package (http://lamda.nju.edu.cn/code_gcForest.ashx). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr. Pang.
"""
import argparse
import numpy as np
import sys
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
sys.path.insert(0, "../lib")

from gcforest.gcforestCS import GCForestCS
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
from gcforest.utils.log_utils import get_logger
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcforestCS Net Model File")
    args = parser.parse_args()
    return args


def get_imdb_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 4
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["part_decay"] = 3
    ca_config["estimators_enlarge"] = True
    ca_config["keep_model_in_mem"] = False
    config["cascadeCS"] = ca_config
    return config


def get_imdb_config_gcForest():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 4
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["keep_model_in_mem"] = False
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    args = parse_args()
    data_name = "imdb"
    if args.model is None:
        config = get_imdb_config()
    else:
        config = load_json(args.model)

    LOGGER = get_logger("examples.demo")
    LOGGER.info("Data set: {} Method: {}".format(data_name,"gcforestCS (gcForest with Confidence Screening)"))

    data = sio.loadmat('../datasets/matData/'+data_name+'.mat')
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"].ravel(), data["y_test"].ravel()
    LOGGER.info("X_train {}, X_test {}".format(X_train.shape, X_test.shape))

    LOGGER.info("==========gcForestCS starts with 100 trees in each forest==========")
    gcCS = GCForestCS(config)
    gcCS.fit_transform(X_train, y_train, X_test, y_test)

    LOGGER.info("==========gcForest with 500 trees in each forest==========")
    config02 = get_imdb_config_gcForest()
    gc_500 = GCForest(config02)
    gc_500.fit_transform(X_train, y_train, X_test, y_test)



