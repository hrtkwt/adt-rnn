import sys
import os
import argparse
import json
import datetime
import logging

import numpy as np
from sklearn.model_selection import KFold

from lib import get_conf, save_conf
from lib.rnn import train

# make save dir from current time
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f'./logs/{now}')

# set logging
logging.basicConfig(
    filename=f'./logs/{now}/train_{now}.log',
    level=logging.INFO
    )
logging.info(f'./logs/{now}/train_{now}.log')

# load parameter
config = get_conf("train")
logging.info(config)
save_conf(config, f'./logs/{now}/feature.json')

feature_date = config["feature_date"]

def expand_dictvalues(a_dict):

    values = list(a_dict.values())

    result = values[0]
    for val in values[1:]:
        result = np.append(result, val, axis=0)
    
    return result

# load dataset
X_train_dict = np.load(f"features/{feature_date}/X_train.npy", allow_pickle=True)[()]
Y_train_dict = np.load(f"features/{feature_date}/Y_train.npy", allow_pickle=True)[()]

X_train_all = expand_dictvalues(X_train_dict)
Y_train_all = expand_dictvalues(Y_train_dict)


kf = KFold(n_splits=config["train_params"]["n_folds"], random_state=0)

for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

    X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
    Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

    train(X_train, X_valid, Y_train, Y_valid, config["train_params"], now, k)
