import random

import numpy as np
import datetime
import logging
import os
import json

from lib import data

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


def train_test_list():
    namelist = data.make_namelist()

    testlist = random.sample(namelist, 4)
    trainlist = list(set(namelist) - set(testlist))

    return trainlist, testlist


def pdd3(arr):
    arr = data.prepro(arr)
    arr = np.abs(arr - arr.mean())
    arr = data.apply_maxmin(arr)

    return arr


trainlist, testlist = train_test_list()

# make train dataset
logging.info("-----train-----")

X_train = dict()
Y_train = dict()

for name in trainlist:
    logging.info(name)

    y = data.load_audio(name, inst="#MIX")
    specs = data.get_spec(y)

    waves = data.smooth_specs(**specs)
    waves["pdd3"] = pdd3(waves["pdd2"])

    #     show_specs(**specs)
    #     show_waves(**waves)

    #     show_spec(specs["a"], "a")
    #     show_wave(waves["pdd3"], "pdd3")

    C_a = specs["a"]
    #    pdd3 = waves["pdd3"][:, np.newaxis]
    targets = data.get_targets(name).T

    C_a = data.make_segments(C_a, 1, 100)
    targets = data.make_segments(targets, 1, 100)[:, -1, :]

    logging.info(C_a.shape)
    logging.info(targets.shape)

    X_train[name] = C_a
    Y_train[name] = targets

# make test dataset
logging.info("-----test-----")

X_test = dict()
Y_test = dict()

for name in testlist:
    logging.info(name)

    y = data.load_audio(name, inst="#MIX")
    specs = data.get_spec(y)

    waves = data.smooth_specs(**specs)
    waves["pdd3"] = pdd3(waves["pdd2"])

    #     show_specs(**specs)
    #     show_waves(**waves)

    #     show_spec(specs["a"], "a")
    #     show_wave(waves["pdd3"], "pdd3")

    C_a = specs["a"]
    targets = data.get_targets(name).T

    #    C_a = data.make_segments(C_a, 1, 100)
    #    targets = data.make_segments(targets, 1, 100)[:,-1,:]

    logging.info(C_a.shape)
    logging.info(targets.shape)

    X_test[name] = C_a
    Y_test[name] = targets


# save
def save(name, arr):

    path = os.path.join("features", now, name)

    np.save(path, arr)


save("X_train", X_train)
save("Y_train", Y_train)

save("X_test", X_test)
save("Y_test", Y_test)
