import os
import datetime
import logging
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

parser = argparse.ArgumentParser()
parser.add_argument("feature")  # 必須の引数を追加
args = parser.parse_args()  # 4. 引数を解析

# params
FEATURE = args.feature

SEED = 2020

NORMALIZE = False

RNN1 = {
    "units": 200,
    "activation": "tanh",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "recurrent_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "recurrent_constraint": None,
    "bias_constraint": None,
    "dropout": 0.0,
    "recurrent_dropout": 0.0,
    "return_sequences": False,
    "return_state": False,
    "go_backwards": False,
    "stateful": False,
    "unroll": False,
}

DENSE = {
    "units": 3,
    "activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
}

ADAM = {
    "learning_rate": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-07,
    "amsgrad": False,
    "name": "Adam",
}

FIT = {
    "batch_size": 10,
    "epochs": 100,
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

VAL_SIZE = 0.2

NFOLDS = -1


def expand(a_dict):

    values = list(a_dict.values())

    result = values[0]
    for val in values[1:]:
        result = np.append(result, val, axis=0)

    return result


def create_model():
    # model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(**RNN1),
            tf.keras.layers.Dense(**DENSE)
        ]
    )

    optimizer = tf.keras.optimizers.Adam(**ADAM)
    loss = tf.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None,
    )

    return model


def train(X_train, X_valid, Y_train, Y_valid, rpath):
    # model
    model = create_model()

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


# make logdir from current time
now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./logs/{now}")

# set logging
logging.basicConfig(filename=f"./logs/{now}/train_{now}.log", level=logging.INFO)

logging.info("-----params-----")
logging.info("---feature")
logging.info(FEATURE)
logging.info("---normalize")
logging.info(NORMALIZE)
logging.info("---rnn")
logging.info(RNN1)
logging.info("---dense")
logging.info(DENSE)
logging.info("---adam")
logging.info(ADAM)
logging.info("---fit")
logging.info(FIT)
logging.info("---val_size")
logging.info(VAL_SIZE)
logging.info("---nfolds")
logging.info(NFOLDS)

# load dataset
X_train_dict = np.load(f"features/{FEATURE}/X_train.npy", allow_pickle=True)[()]
Y_train_dict = np.load(f"features/{FEATURE}/Y_train.npy", allow_pickle=True)[()]

X_train_all = expand(X_train_dict)
Y_train_all = expand(Y_train_dict)


def apply_zscore(x):
    return (x - x.mean()) / x.std()


if NORMALIZE:
    X_train_all = apply_zscore(X_train_all)

if NFOLDS == -1:

    rpath = f"{now}/{-1}"

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train_all, Y_train_all, test_size=VAL_SIZE, random_state=SEED
    )

    model = train(X_train, X_valid, Y_train, Y_valid, rpath)

else:
    kf = KFold(n_splits=NFOLDS, random_state=SEED)
    for k, (train_index, valid_index) in enumerate(kf.split(X_train_all)):

        rpath = f"{now}/{k+1}"

        X_train, X_valid = X_train_all[train_index], X_train_all[valid_index]
        Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

        model = train(X_train, X_valid, Y_train, Y_valid, rpath)
