import random

import numpy as np
import datetime
import logging
import os

import librosa

from lib.data import get_targets

# params
SEED = 1234

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

SEG = {"seg_width": 100, "seg_step": 10}

MODE = "gamma"

NAME = "D_gamma"


def deviation(x, foward=False):
    x_d = np.zeros(x.shape, dtype=x.dtype)

    for i in range(1, len(x_d)):
        x_d[i] = x[i] - x[i - 1]

    return x_d


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


def get_func_feature(mode):
    if mode == "m":
        return feature_m
    elif mode == "m_md":
        return feature_m_md
    elif mode == "m_acd":
        return feature_m_acd
    elif mode == "acd":
        return feature_acd
    elif mode == "gamma":
        return feature_gamma
    else:
        print("load_func_err")
        exit(0)


def feature_m(audioname):
    # get audiopath from audioname
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    # abs
    X_m = np.abs(X_c)

    X = X_m
    Y = get_targets(name).T

    return X, Y


def feature_m_md(audioname):
    # get audiopath from audioname
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    X_m = np.abs(X_c)
    X_md = deviation(X_m)

    X = np.hstack((X_m, X_md))
    Y = get_targets(name).T

    return X, Y


def feature_m_acd(audioname):
    # get audiopath from audioname
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    X_m = np.abs(X_c)
    X_cd = deviation(X_c)
    X_acd = np.abs(X_cd)

    X = np.hstack((X_m, X_acd))
    Y = get_targets(name).T

    return X, Y


def feature_acd(audioname):
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    X_cd = deviation(X_c)
    X_acd = np.abs(X_cd)

    X = X_acd
    Y = get_targets(name).T

    return X, Y


def feature_gamma(audioname):
    def princarg(p):
        return ((p + np.pi) % -2 * np.pi) + np.pi

    audiopath = get_audiopath(audioname, INST)
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T
    X_m = np.abs(X_c)
    X_p = np.angle(X_c)
    X_up = np.unwrap(X_p)

    X_gamma = np.abs(X_m)

    for t in range(2, X_gamma.shape[1]):
        phi_target = princarg(2 * X_up[:, t - 1] - X_up[:, t - 2])
        r_target = np.abs(X_m[:, t - 1])

        phi_current = np.angle(X_c[:, t])
        r_current = np.abs(X_c[:, t])

        X_gamma[:, t] = np.sqrt(
            r_target ** 2
            + r_current ** 2
            - 2 * r_target * r_current * np.cos(phi_target - phi_current)
        )

    X = X_gamma
    Y = get_targets(name).T

    return X, Y


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
if True:
    now = NAME
os.makedirs(f"./features/{now}")

# set logging
logging.basicConfig(filename=f"./features/{now}/feature_{now}.log", level=logging.INFO)

# save params
logging.info("----params-----")
logging.info("---mode")
logging.info(MODE)
logging.info("---seed")
logging.info(SEED)
logging.info("---inst")
logging.info(INST)
logging.info("---audio_params")
logging.info(AUDIO)
logging.info("---stft_params")
logging.info(STFT)
logging.info("---seg_params")
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

    feature = get_func_feature(MODE)
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
