import os
import datetime
import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

# params
FEATURE_DATE = "20191202-143743"

SEED = 0

RNN = {
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
    "activation": None,
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
    "epochs": 20,
    "verbose": 1,
    "shuffle": True,
    "class_weight": None,
    "sample_weight": None,
    "initial_epoch": 1,
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
        [tf.keras.layers.SimpleRNN(**RNN), tf.keras.layers.Dense(**DENSE)]
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
    cp_path = f"cp/{rpath}" + "/cp-{epoch:04d}"
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
os.makedirs(f"./cp/{now}")

# set logging
logging.basicConfig(filename=f"./cp/{now}/train_{now}.log", level=logging.INFO)

logging.info("-----params-----")
logging.info("rnn")
logging.info(RNN)
logging.info("dense")
logging.info(DENSE)
logging.info("adam")
logging.info(ADAM)
logging.info("fit")
logging.info(FIT)
logging.info("val_size")
logging.info(VAL_SIZE)
logging.info("nfolds")
logging.info(NFOLDS)

# load dataset
X_train_dict = np.load(f"features/{FEATURE_DATE}/X_train.npy", allow_pickle=True)[()]
Y_train_dict = np.load(f"features/{FEATURE_DATE}/Y_train.npy", allow_pickle=True)[()]

X_train_all = expand(X_train_dict)
Y_train_all = expand(Y_train_dict)

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
