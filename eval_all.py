import os
import datetime
import logging

import numpy as np
import pandas as pd

from lib.evalutil import peaks, accuracies, get_result_table
from lib.plot import save_spec, save_Y
import tensorflow as tf

# from lib.models import select_model

FEATURE = "E_gamma"
MODEL = "RNN1"
FOLD = -1
EPOCH = 10
PEAK = {
    "pre_max": 3,
    "post_max": 3,
    "pre_avg": 3,
    "post_avg": 3,
    "delta": "all",
    "wait": 2,
}
TOLERANCE = {"pre_tolerance": 3, "post_tolerance": 3}
FIG = False
DIRNAME = f"{FEATURE}_{MODEL}"

now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./results/{DIRNAME}/fig")

# set logging
logging.basicConfig(
    filename=f"./results/{DIRNAME}/eval_{now}.log", level=logging.INFO
)
logging.info("-----params")
items = list(globals().items())
for (symbol, value) in items:
    if symbol.isupper():
        logging.info(f"---{symbol}")
        logging.info(value)

# load model
cp_path = f"logs/{DIRNAME}/{FOLD}/cp-{EPOCH:04d}"
model = tf.keras.models.load_model(cp_path)

# load test data
X_test_dict = np.load(f"features/{FEATURE}/X_test.npy", allow_pickle=True)[()]
Y_test_dict = np.load(f"features/{FEATURE}/Y_test.npy", allow_pickle=True)[()]


def eval(name):

    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]

    # pred
    Y_pred = model(X_test[np.newaxis, :, :])[0].numpy()

    # peak_picking
    Y_peak = peaks(Y_pred, **PEAK)

    # serialize image
    if FIG is True:
        save_spec(X_test, f"{DIRNAME}", name, "X_test")
        save_Y(Y_pred, f"{DIRNAME}", name, "pred")
        save_Y(Y_peak, f"{DIRNAME}", name, "peak")
        save_Y(Y_test, f"{DIRNAME}", name, "gt")

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
    result_table.to_csv(f"results/{DIRNAME}/{FEATURE}_{MODEL}.csv")

else:
    test_names = X_test_dict.keys()

    result_table = get_result_table(test_names)

    for name in test_names:
        result = eval(name)
        result_table.loc[name] = result

    print(result_table)

    result_table.to_csv(f"results/{DIRNAME}/{FEATURE}_{MODEL}.csv")
