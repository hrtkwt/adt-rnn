import os

import datetime
import logging

import argparse

import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split, KFold
from lib.models import select_model

if True:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature")  # 必須の引数を追加
    args = parser.parse_args()  # 4. 引数を解析

# params
FEATURE = args.feature
SEED = 2000
MODEL = "RNN1"
FIT = {
    "batch_size": 10,
    "epochs": 12,
    "verbose": 2,
    "shuffle": True,
    "class_weight": None,
    "sample_weight": None,
    "initial_epoch": 0,
    "steps_per_epoch": None,
    "validation_steps": None,
    "validation_freq": 1,
    "max_queue_size": 10,
    "workers": 1,
    "use_multiprocessing": False,
}
VALSIZE = 0.2
NFOLDS = -1
DIRNAME = f"{FEATURE}_{MODEL}"


def expand(a_dict):

    values = list(a_dict.values())

    result = values[0]
    for val in values[1:]:
        result = np.append(result, val, axis=0)

    return result


def train(X_train, X_valid, Y_train, Y_valid, rpath):
    model = select_model(MODEL)

    # set callbacks
    # tb_callback
    logdir = f"logs/{rpath}"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # cp_callback
    cp_path = f"logs/{rpath}" + "/cp-{epoch:04d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_path, save_weights_only=False, verbose=1
    )

    # train
    model.fit(
        x=X_train,
        y=Y_train,
        callbacks=[tb_callback, cp_callback],
        validation_data=(X_valid, Y_valid),
        **FIT,
    )

    return model


now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./logs/{DIRNAME}")

# set logging
logging.basicConfig(filename=f"./logs/{DIRNAME}/train_{now}.log", level=logging.INFO)
logging.info("-----params")
items = list(globals().items())
for (symbol, value) in items:
    if symbol.isupper():
        logging.info(f"---{symbol}")
        logging.info(value)

# load dataset
X_train_dict = np.load(f"features/{FEATURE}/X_train.npy", allow_pickle=True)[()]
Y_train_dict = np.load(f"features/{FEATURE}/Y_train.npy", allow_pickle=True)[()]

X_train = expand(X_train_dict)
Y_train = expand(Y_train_dict)

X_valid_dict = np.load(f"features/{FEATURE}/X_valid.npy", allow_pickle=True)[()]
Y_valid_dict = np.load(f"features/{FEATURE}/Y_valid.npy", allow_pickle=True)[()]

X_valid = expand(X_valid_dict)
Y_valid = expand(Y_valid_dict)

if NFOLDS == -1:

    rpath = f"{DIRNAME}/{-1}"
    model = train(X_train, X_valid, Y_train, Y_valid, rpath)

# else:
#     kf = KFold(n_splits=NFOLDS, random_state=SEED)
#     for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

#         rpath = f"{DIRNAME}/{k+1}"

#         X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
#         Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

#         model = train(X_train, X_valid, Y_train, Y_valid, rpath)
