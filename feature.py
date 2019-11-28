import random
import argparse
import json
import numpy as np
import datetime

from utils import data


# load parameter
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/feature.json')
options = parser.parse_args()
config = json.load(open(options.config))

print(config)

def train_test_list():
    namelist = data.make_namelist()

    testlist = random.sample(namelist, 4)
    trainlist = list(set(namelist) - set(testlist))

    return trainlist, testlist

trainlist, testlist = train_test_list()

X_a_train = []
target_train = []

for name in trainlist:
    print(name)
    
    y = data.load_audio(name, inst="#MIX")
    specs = data.get_spec(y)
    
    waves = data.smooth_specs(**specs)
    waves["pdd2"] = data.prepro(waves["pdd2"])
    waves["pdd3"] = np.abs(waves["pdd2"] - waves["pdd2"].mean())
    waves["pdd3"] = data.apply_maxmin(waves["pdd3"])
    
#     show_specs(**specs)
#     show_waves(**waves)

#     show_spec(specs["a"], "a")
#     show_wave(waves["pdd3"], "pdd3")
    
    C_a = specs["a"]
#    pdd3 = waves["pdd3"][:, np.newaxis]
    targets = data.get_targets(name).T
    
    C_a = data.make_segments(C_a, 1, 100)
    targets = data.make_segments(targets, 1, 100)[:,-1,:]
    
    print(C_a.shape)
    print(targets.shape)
    
    X_a_train.extend(C_a)
    target_train.extend(targets)

X_a_train = np.array(X_a_train)
target_train = np.array(target_train)

print(X_a_train.shape)
print(target_train.shape)

X_a_test = []
target_test = []

for name in testlist:
    print(name)
    
    y = data.load_audio(name, inst="#MIX")
    specs = data.get_spec(y)
    
    waves = data.smooth_specs(**specs)
    waves["pdd2"] = data.prepro(waves["pdd2"])
    waves["pdd3"] = np.abs(waves["pdd2"] - waves["pdd2"].mean())
    waves["pdd3"] = data.apply_maxmin(waves["pdd3"])
    
#     show_specs(**specs)
#     show_waves(**waves)

#     show_spec(specs["a"], "a")
#     show_wave(waves["pdd3"], "pdd3")
    
    C_a = specs["a"]
    targets = data.get_targets(name).T
    
#    C_a = data.make_segments(C_a, 1, 100)
#    targets = data.make_segments(targets, 1, 100)[:,-1,:]
    
    print(C_a.shape)
    print(targets.shape)
    
    X_a_test.extend(C_a)
    target_test.extend(targets)
    
X_a_test = np.array(X_a_test)
target_test = np.array(target_test)

print(X_a_test.shape)
print(target_test.shape)

now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(now)

