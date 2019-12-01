import os
import json
import datetime
import logging

import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import KFold

from lib.rnn import train

# make save dir from current time
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f"./logs/{now}")

# set logging
logpath = f"./logs/{now}/train_{now}.log"
logging.basicConfig(filename=logpath, level=logging.INFO)
logging.info(logpath)

# load parameter
with open("configs/train.json", "r") as f:
    config = json.load(f)
    logging.info(config)

with open(f"./logs/{now}/feature.json", "w") as f:
    json.dump(config, f)

feature_date = config["feature_date"]


def expand_dictvalues(a_dict):

    values = list(a_dict.values())

    result = values[0]
    for val in values[1:]:
        result = np.append(result, val, axis=0)

    return result


# load dataset
loadpath = f"features/{feature_date}/X_train.npy"
X_train_dict = np.load(loadpath, allow_pickle=True)[()]
Y_train_dict = np.load(loadpath, allow_pickle=True)[()]

X_train_all = expand_dictvalues(X_train_dict)
Y_train_all = expand_dictvalues(Y_train_dict)


X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train_all, Y_train_all, test_size=0.2, random_state=0
)

model = train(X_train, X_valid, Y_train, Y_valid, config["train_params"], now, 1)

# cv

# kf = KFold(n_splits=config["train_params"]["n_folds"], random_state=0)
# for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

#     X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
#     Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

#     model = train(X_train, X_valid, Y_train, Y_valid, config["train_params"], now, k)
