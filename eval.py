import os
import datetime
import logging

import numpy as np

from lib.evalutil import peaks, accuracies, get_result_table, get_weights
from lib.plot import save_spec, save_Y

FEATURE = "m_10"
TRAIN = "base_m_10"
FOLD = -1
EPOCH = 15

PEAK = {
    "pre_max": 3,
    "post_max": 3,
    "pre_avg": 3,
    "post_avg": 3,
    "delta": 0.05,
    "wait": 2,
}

TOLERANCE = {"pre_tolerance": 3, "post_tolerance": 3}

# make logdir from current time
now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./results/{now}/fig")

# set logging
logging.basicConfig(filename=f"./results/{now}/train_{now}.log", level=logging.INFO)

logging.info("-----params-----")
logging.info("--feature")
logging.info(FEATURE)
logging.info("--train")
logging.info(TRAIN)
logging.info("--fold")
logging.info(FOLD)
logging.info("--epoch")
logging.info(EPOCH)
logging.info("--peak")
logging.info(PEAK)
logging.info("--tolerance")
logging.info(TOLERANCE)

# make cp-path
cp_path = f"logs/{TRAIN}/{FOLD}/cp-{EPOCH:04d}"

# get weights
weights = get_weights(cp_path)

# load test data
X_test_dict = np.load(f"features/{FEATURE}/X_test.npy", allow_pickle=True)[()]
Y_test_dict = np.load(f"features/{FEATURE}/Y_test.npy", allow_pickle=True)[()]


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

    # pred
    Y_pred = pred(X_test, *weights)

    # peak_picking
    Y_peak = peaks(Y_pred, **PEAK)

    # serialize image
    if True:
        save_spec(X_test, now, name, "X_test")
        save_Y(Y_pred, now, name, "pred")
        save_Y(Y_peak, now, name, "peak")
        save_Y(Y_test, now, name, "gt")

    # eval
    result = accuracies(Y_test, Y_peak, **TOLERANCE)

    return result


test_names = X_test_dict.keys()

result_table = get_result_table(test_names)

for name in test_names:
    result = eval(name)
    result_table.loc[name] = result

print(result_table)

result_table.to_csv(f"results/{now}/result_table.csv")
