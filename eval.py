import sys
import numpy as np

from lib import get_conf
from lib.data import load_test_data, load_test_target
from lib.rnn import load_weights, pred
from lib.evalutil import peaks, accuracies

# load model
config = get_conf("eval")
print(config)

weights = load_weights(**config["weights"])

feature_date = config["feature_date"]


# load test data
X_test_dict = np.load(f"features/{feature_date}/X_test.npy", allow_pickle=True)[()]
Y_test_dict = np.load(f"features/{feature_date}/Y_test.npy", allow_pickle=True)[()]

print(X_test_dict.keys())
print(Y_test_dict.keys())

X_test = X_test_dict["RealDrum01_01"]
Y_test = X_test_dict["RealDrum01_01"]

# pred
Y_pred = pred(X_test, weights)

# peak_picking
Y_peak = peaks(Y_pred, **config["peak_params"])
print(Y_test.shape)
print(Y_peak.shape)
# eval
result = accuracies(Y_test, Y_peak, **config["metrics_params"])

print(result)

