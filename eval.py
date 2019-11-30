import tensorflow as tf
from lib.rnn import load_model
from lib import get_conf

from lib.data import load_test_data
from lib.data import load_test_target
# load model

config = get_conf("eval")
print(config)

model = load_model("20191130-172056", 0, 2)

model.summary()

for w in model.weights:
    print(w.shape)

# load test

X_test = load_test_data(date="pesu")
Y_test = load_test_target(date="pesu")

print(X_test.shape)
print(Y_test.shape)

Y_pred = model.predict(X_test)
print(Y_pred.shape)

# foward propagation

# peak_picking

# 