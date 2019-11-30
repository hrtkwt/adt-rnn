import sys
import numpy as np

from lib import get_conf
from lib.data import load_test_data, load_test_target
from lib.rnn import load_weights, pred
from lib.evalutil import peaks, accuracies

# load model

config = get_conf("eval")
print(config)

weights = load_weights(**config["weights_param"])

X_test = load_test_data(**config["feature_param"])
Y_test = load_test_target(**config["feature_param"])

print(X_test.shape)
print(Y_test.shape)

# pred
Y_pred = pred(X_test, weights)

# peak_picking
Y_peak = peaks(Y_pred, **config["peak_params"])

metrics_params = {"pre_tolerance": 3, "post_tolerance": 3}
result = accuracies(Y_test, Y_peak, **metrics_params)

print(result)

