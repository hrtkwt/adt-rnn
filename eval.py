import json
import os
import datetime

import numpy as np

from lib.rnn import pred
from lib.evalutil import peaks, accuracies, get_result_table, get_weights


def open_and_save(openpath, savepath):
    with open(savepath, "w") as fs:
        with open(openpath, "r") as fo:
            cnf = json.load(fo)
        json.dump(cnf, fs)


now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f"./results/{now}")

with open("configs/eval.json", "r") as f:
    config = json.load(f)

with open(f"./results/{now}/eval.json", "w") as f:
    json.dump(config, f)


feature_date = config["feature_date"]

open_and_save(
    openpath=f"./features/{feature_date}/feature.json",
    savepath=f"./results/{now}/feature.json",
)

train_date = config["train_date"]
open_and_save(
    openpath=f"./logs/{train_date}/train.json", savepath=f"./results/{now}/train.json"
)

# make cp-path
train_date = config["train_date"]
fold = config["fold"]
epoch = config["epoch"]

cp_path = f"cp/{train_date}/{fold}/cp-{epoch:04d}"

# get weights
weights = get_weights(cp_path)

# load test data
X_test_dict = np.load(f"features/{feature_date}/X_test.npy", allow_pickle=True)[()]
Y_test_dict = np.load(f"features/{feature_date}/Y_test.npy", allow_pickle=True)[()]


def eval(name):

    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]

    # pred
    Y_pred = pred(X_test, *weights)

    # peak_picking
    Y_peak = peaks(Y_pred, **config["peak_params"])

    # eval
    result = accuracies(Y_test, Y_peak, **config["metrics_params"])

    return result


test_names = X_test_dict.keys()

result_table = get_result_table(test_names)

for name in test_names:
    result = eval(name)
    result_table.loc[name] = result

print(result_table)

result_table.to_csv(f"results/{now}/result.csv")
