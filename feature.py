import random

import numpy as np
import datetime
import logging
import os

import librosa

from lib.data import get_targets

# params
SEED = 0

INST = "#MIX"

AUDIO = {
    "sr": 44100,
    "mono": True,
    "offset": 0.0,
    "duration": None,
    "dtype": np.float32,
    "res_type": "kaiser_best",
}

STFT = {
    "n_fft": 2048,
    "hop_length": 512,
    "win_length": None,
    "window": "hann",
    "center": True,
    "dtype": np.complex64,
    "pad_mode": "reflect",
}

SEG = {"seg_width": 100, "seg_step": 1}


def make_namelist():
    namelist = []
    for i in range(20):
        namelist.append("RealDrum01_{:02d}".format(i))

    return namelist


def get_audiopath(audioname, inst):
    if inst == "#MIX":
        path = "/home/cs181004/data/SMT_DRUMS/MIX"
        audiopath = os.path.join(path, audioname + "#MIX.wav")
    else:
        path = "/home/cs181004/data/SMT_DRUMS/XX#train"
        audiopath = os.path.join(path, audioname + inst + "#train.wav")

    return audiopath


def feature(audioname):
    # get audiopath from audioname
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    # abs
    X_abs = np.abs(X_c)
    Y = get_targets(name).T

    return X_abs, Y


def make_segments(arr, seg_width, seg_step):
    t, n_bins = arr.shape

    # padding (might have an issue)
    pad = np.zeros((seg_width, n_bins))
    arr = np.vstack([pad, arr])

    segments = []

    k = 0
    while k < t:
        segments.append(arr[k : k + seg_width])
        k += seg_step

    segments = np.array(segments, dtype=arr.dtype)

    return segments


# make save dir from current time
now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./features/{now}")

# set logging
logging.basicConfig(filename=f"./features/{now}/feature_{now}.log", level=logging.INFO)

# save params
logging.info("----params-----")
logging.info("seed")
logging.info(SEED)
logging.info("inst")
logging.info(INST)
logging.info("audio_params")
logging.info(AUDIO)
logging.info("stft_params")
logging.info(STFT)
logging.info("seg_params")
logging.info(SEG)

# make train test namelist
random.seed(SEED)
namelist = make_namelist()

testlist = random.sample(namelist, 4)
trainlist = list(set(namelist) - set(testlist))

# make train dataset
logging.info("-----train-----")

X_train = dict()
Y_train = dict()

for name in trainlist:
    logging.info(name)

    X, Y = feature(name)

    # make segments
    X = make_segments(X, **SEG)
    Y = make_segments(Y, **SEG)[:, -1, :]

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
