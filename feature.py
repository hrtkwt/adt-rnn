import random

import numpy as np
import datetime
import logging
import os
import json

from lib.data import (
    make_namelist,
    load_audio,
    get_specs,
    get_targets,
    smooth_specs,
    make_segments,
    pdd3,
)


# make save dir from current time
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f"./features/{now}")

# set logging
logging.basicConfig(filename=f"./features/{now}/feature_{now}.log", level=logging.INFO)
logging.info(f"./features/{now}/feature_{now}.log")

# load parameter
with open("configs/feature.json") as f:
    config = json.load(f)
logging.info(config)

# make train test namelist
namelist = make_namelist()
testlist = random.sample(namelist, 4)
trainlist = list(set(namelist) - set(testlist))


def feature(name):

    y = load_audio(name, inst="#MIX")
    specs = get_specs(y)

    waves = smooth_specs(**specs)
    waves["pdd3"] = pdd3(waves["pdd2"])

    targets = get_targets(name).T

    return specs["a"], targets


# make train dataset
logging.info("-----train-----")

X_train = dict()
Y_train = dict()

for name in trainlist:
    logging.info(name)

    X, Y = feature(name)
    X = make_segments(X, 100, 1)
    Y = make_segments(Y, 100, 1)[:, -1, :]

    logging.info(X.shape)
    logging.info(Y.shape)

    X_train[name] = X
    Y_train[name] = Y

# make test dataset
logging.info("-----test-----")

X_test = dict()
Y_test = dict()

for name in testlist:
    logging.info(name)

    X, Y = feature(name)

    logging.info(X.shape)
    logging.info(Y.shape)

    X_test[name] = X
    Y_test[name] = Y


# save
def save(name, arr):

    path = os.path.join("features", now, name)
    np.save(path, arr)


save("X_train", X_train)
save("Y_train", Y_train)

save("X_test", X_test)
save("Y_test", Y_test)
