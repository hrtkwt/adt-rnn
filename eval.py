import json

import numpy as np

from lib.rnn import load_weights, pred
from lib.evalutil import peaks, accuracies

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

X_test = X_test_dict["RealDrum01_01"]
Y_test = Y_test_dict["RealDrum01_01"]

# pred
Y_pred = pred(X_test, *weights)

# peak_picking
Y_peak = peaks(Y_pred, **config["peak_params"])

# eval
result = accuracies(Y_test, Y_peak, **config["metrics_params"])

print(result)
