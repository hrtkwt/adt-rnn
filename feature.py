import random

import numpy as np
import datetime
import logging
import os
import sys

sys.path.append("..")

from utils import data

print(type(data))

sys.exit()



# make save dir from current time
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f'./feature/{now}')

# set logging
logging.basicConfig(
    filename=f'./feature/{now}/log_{now}.log',
    level=logging.INFO
    )
logging.info(f'./feature/{now}/log_{now}.log')


# load parameter
config = get_conf("feature")
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

X_a_train = dict()
target_train = dict()

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
    targets = data.make_segments(targets, 1, 100)[:,-1,:]
    
    logging.info(C_a.shape)
    logging.info(targets.shape)
    
    X_a_train[name] = C_a
    target_train[name] = targets

print(X_a_train)
print(target_train)


# make test dataset

X_a_test = dict()
target_test = dict()

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
    
    X_a_test[name] = C_a
    target_test[name] = targets
    
print(X_a_test)
print(target_test)


# save
def save(name, arr):

    path = os.path.join("features", now, name)

    np.save(path, arr)


save("X_a_train", X_a_train)
save("target_train", target_train)

save("X_a_test", X_a_test)
save("target_test", target_test)
