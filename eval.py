import sys
import numpy as np

from lib import get_conf
from lib.data import load_test_data, load_test_target
from lib.rnn import load_weights, pred
from lib.evalutil import peaks, accuracies

# load model
config = get_conf("eval")
print(config)

weights = load_weights(**config["weights_params"])

# load test data
X_test = load_test_data(**config["feature_params"])
Y_test = load_test_target(**config["feature_params"])

# pred
Y_pred = pred(X_test, weights)

# peak_picking
Y_peak = peaks(Y_pred, **config["peak_params"])

# eval
result = accuracies(Y_test, Y_peak, **config["metrics_params"])

print(result)

