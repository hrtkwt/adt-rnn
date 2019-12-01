import json

import numpy as np

from lib.rnn import load_weights, pred
from lib.evalutil import peaks, accuracies, get_result_table

# load model
with open("configs/eval.json") as f:
    config = json.load(f)

weights = load_weights(**config["weights"])

feature_date = config["feature_date"]

# load test data
X_test_dict = np.load(f"features/{feature_date}/X_test.npy", allow_pickle=True)
X_test_dict = X_test_dict[()]
Y_test_dict = np.load(f"features/{feature_date}/Y_test.npy", allow_pickle=True)
Y_test_dict = Y_test_dict[()]


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

print(result_table)

for name in test_names:
    result = eval(name)
    result_table.loc[name] = result

print(result_table)
