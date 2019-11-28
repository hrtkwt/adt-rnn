import sys
import argparse
import json

from sklearn.model_selection import KFold

from utils.data import load_train_data, load_train_target
from models.rnn import train


# load parameter
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/train.json')
options = parser.parse_args()
config = json.load(open(options.config))

# load dataset
X_train_all = load_train_data(date=config["date"])
Y_train_all = load_train_target(date=config["date"])


kf = KFold(n_splits=3, random_state=0)

for train_index, valid_index in kf.split(X_train_all):

    X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
    Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

    train(X_train, X_valid, Y_train, Y_valid, config["params"])
