import sys
import os
import argparse
import json
import datetime
import logging

from sklearn.model_selection import KFold

from lib.data import load_train_data, load_train_target
from lib.rnn import train
from lib import get_conf

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

# load dataset
X_train_all = load_train_data(date=config["feature_date"])
Y_train_all = load_train_target(date=config["feature_date"])


kf = KFold(n_splits=3, random_state=0)

for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

    X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
    Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

    train(X_train, X_valid, Y_train, Y_valid, config["train_params"], now, k)
