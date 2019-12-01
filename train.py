import os
import json
import datetime
import logging

import numpy as np
from sklearn.model_selection import train_test_split, KFold

from lib.rnn import train

# make logdir from current time
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f"./logs/{now}")

# logging setting
logging.basicConfig(filename=f"./logs/{now}/train_{now}.log", level=logging.INFO)

# load config
with open("configs/train.json", "r") as f:
    config = json.load(f)

with open(f"./logs/{now}/train.json", "w") as f:
    json.dump(config, f)

logging.info(config)

# load dataset
feature_date = config["feature_date"]

X_train_dict = np.load(f"features/{feature_date}/X_train.npy", allow_pickle=True)[()]
Y_train_dict = np.load(f"features/{feature_date}/Y_train.npy", allow_pickle=True)[()]


def expand_dictvalues(a_dict):

    values = list(a_dict.values())

    result = values[0]
    for val in values[1:]:
        result = np.append(result, val, axis=0)

    return result


X_train_all = expand_dictvalues(X_train_dict)
Y_train_all = expand_dictvalues(Y_train_dict)

nfolds = config["train_params"]["n_folds"]

if nfolds == -1:

    rpath = f"{now}/{-1}"

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train_all, Y_train_all, test_size=0.2, random_state=0
    )

    model = train(X_train, X_valid, Y_train, Y_valid, config["train_params"], rpath)

else:
    kf = KFold(n_splits=nfolds, random_state=0)
    for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

        rpath = f"{now}/{k+1}"

        X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
        Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

        model = train(X_train, X_valid, Y_train, Y_valid, config["train_params"], rpath)
