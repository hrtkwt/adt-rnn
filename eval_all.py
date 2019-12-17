import os
import datetime
import logging

import numpy as np
import pandas as pd

from lib.evalutil import peaks, accuracies, get_result_table, get_weights
from lib.plot import save_spec, save_Y

FEATURE = "C_m_acd_10"
TRAIN = "C_n_hidden_400_m_acd_10"
FOLD = -1
EPOCH = 77


PEAK = {
    "pre_max": 3,
    "post_max": 3,
    "pre_avg": 3,
    "post_avg": 3,
    "delta": "all",
    "wait": 2,
}

NORMALIZE = False

TOLERANCE = {"pre_tolerance": 3, "post_tolerance": 3}

FIG = False

# make logdir from current time
now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./results/{TRAIN}/fig")

# set logging
logging.basicConfig(filename=f"./results/{TRAIN}/eval_{now}.log", level=logging.INFO)
logging.info("-----params")
items = list(globals().items())
for (symbol, value) in items:
    if symbol.isupper():
        logging.info(f"---{symbol}")
        logging.info(value)

# make cp-path
cp_path = f"logs/{TRAIN}/{FOLD}/cp-{EPOCH:04d}"

# get weights
weights = get_weights(cp_path)

# load test data
X_test_dict = np.load(f"features/{FEATURE}/X_test.npy", allow_pickle=True)[()]
Y_test_dict = np.load(f"features/{FEATURE}/Y_test.npy", allow_pickle=True)[()]


def apply_zscore(x):
    return (x - x.mean()) / x.std()


def pred(X, Wh, Wr, bh, Wo, bo):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def feed_forward(x, h_):
        zh = np.dot(x, Wh) + np.dot(h_, Wr) + bh
        h = sigmoid(zh)

        zo = np.dot(h, Wo) + bo
        y = sigmoid(zo)

        return y, h

    x0 = X[0]
    h0 = np.zeros(bh.shape)

    Y, h = feed_forward(x0, h0)

    for x in X[1:]:
        y, h = feed_forward(x, h)
        Y = np.vstack((Y, y))

    return Y


def eval(name):

    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]

    if NORMALIZE is True:
        X_test = apply_zscore(X_test)

    # pred
    Y_pred = pred(X_test, *weights)

    # peak_picking
    Y_peak = peaks(Y_pred, **PEAK)

    # serialize image
    if FIG is True:
        save_spec(X_test, now, name, "X_test")
        save_Y(Y_pred, now, name, "pred")
        save_Y(Y_peak, now, name, "peak")
        save_Y(Y_test, now, name, "gt")

    # eval
    result = accuracies(Y_test, Y_peak, **TOLERANCE)

    return result


test_names = X_test_dict.keys()


def eval_at(thres):
    PEAK["delta"] = thres

    result_table = get_result_table(test_names)
    result_table = pd.concat([result_table], keys=[thres], names=["thres"])
    for name in test_names:
        result = eval(name)
        result_table.loc[(thres, name)] = result

    return result_table.describe().loc["mean"]


if PEAK["delta"] == "all":
    thres_list = [round(0.05 * i, 2) for i in range(1, 20)]

    thres = thres_list[0]
    result_table = get_result_table(["dummy"])
    result_table.loc[(thres)] = eval_at(thres)
    result_table = result_table.drop(["dummy"], axis=0)

    for thres in thres_list[1:]:
        result_table.loc[(thres)] = eval_at(thres)

    print(result_table)
    result_table.to_csv(f"results/{TRAIN}/{TRAIN}.csv")

else:
    test_names = X_test_dict.keys()

    result_table = get_result_table(test_names)

    for name in test_names:
        result = eval(name)
        result_table.loc[name] = result

    print(result_table)

    result_table.to_csv(f"results/{TRAIN}/{TRAIN}.csv")
