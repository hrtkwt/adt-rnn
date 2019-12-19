import random
import argparse
import numpy as np
import datetime
import logging
import os

import librosa

from lib.data import get_targets

if True:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature")
    args = parser.parse_args()

# params
SEED = 12181135
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
SEG = {"seg_width": 100, "seg_step": 50}
FEATURE = args.feature
NORMALIZE = "z"
AUDIOTYPE = "r"
PREFIX = "M"
LAST = False
NVALID = 3
NTEST = 3
DIRNAME = f"{PREFIX}_{FEATURE}"


def deviation(x, foward=False):
    x_d = np.zeros(x.shape, dtype=x.dtype)

    for i in range(1, len(x_d)):
        x_d[i] = x[i] - x[i - 1]

    return x_d


def make_namelist(mode):
    namelist = []

    if "r" in mode:
        for i in range(20):
            namelist.append("RealDrum01_{:02d}".format(i))
    if "t" in mode:
        for i in range(10):
            namelist.append("TechnoDrum01_{index:02}".format(index=i))
        for i in range(4):
            namelist.append("TechnoDrum02_{index:02}".format(index=i))

    if "w" in mode:
        for i in range(10):
            namelist.append("WaveDrum01_{index:02}".format(index=i))
        for i in range(1, 61):
            namelist.append("WaveDrum02_{index:02}".format(index=i))

    return namelist


def get_audiopath(audioname, inst):
    if inst == "#MIX":
        path = "/home/cs181004/data/SMT_DRUMS/MIX"
        audiopath = os.path.join(path, audioname + "#MIX.wav")
    else:
        path = "/home/cs181004/data/SMT_DRUMS/XX#train"
        audiopath = os.path.join(path, audioname + inst + "#train.wav")

    return audiopath


def select_func_feature(mode):
    if mode == "m":
        return feature_m
    elif mode == "md":
        return feature_md
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


def feature_md(audioname):
    # get audiopath from audioname
    audiopath = get_audiopath(audioname, INST)
    y, _ = librosa.load(path=audiopath, **AUDIO)
    X_c = librosa.core.stft(y=y, **STFT).T

    X_m = np.abs(X_c)
    X_md = deviation(X_m)

    X = X_md
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


def select_func_normalize(mode):
    if mode == "z":
        return apply_zscore
    if mode == "idx":
        return lambda x: x
    else:
        exit(1)


def apply_zscore(x):
    return (x - x.mean()) / x.std()


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


now = datetime.datetime.now().strftime("%m%d-%H%M%S")
os.makedirs(f"./features/{DIRNAME}")

# set logging
logging.basicConfig(
    filename=f"./features/{DIRNAME}/feature_{now}.log", level=logging.INFO
)

# save params
logging.basicConfig(
    filename=f"./features/{DIRNAME}/feature_{now}.log", level=logging.INFO
)
logging.info("-----params")
items = list(globals().items())
for (symbol, value) in items:
    if symbol.isupper():
        logging.info(f"---{symbol}")
        logging.info(value)


# make train test namelist
namelist = sorted(make_namelist(AUDIOTYPE))

random.seed(SEED)
testlist = sorted(random.sample(namelist, NTEST))
trainlist = sorted(list(set(namelist) - set(testlist)))

random.seed(SEED)
validlist = sorted(random.sample(trainlist, NVALID))
trainlist = sorted(list(set(trainlist) - set(validlist)))

# select feature and normalize func
feature = select_func_feature(FEATURE)
normalize = select_func_normalize(NORMALIZE)


# make train dataset
logging.info("-----train-----")

X_train = dict()
Y_train = dict()

for name in trainlist:
    logging.info(name)

    X, Y = feature(name)
    X = normalize(X)

    # make segments
    X = make_segments(X, **SEG)
    Y = make_segments(Y, **SEG)
    if LAST:
        Y = Y[:, -1, :]

    logging.info(X.shape)
    logging.info(Y.shape)

    X_train[name] = X
    Y_train[name] = Y

# make train dataset
logging.info("-----valid-----")

X_valid = dict()
Y_valid = dict()

for name in validlist:
    logging.info(name)

    X, Y = feature(name)
    X = normalize(X)

    # make segments
    X = make_segments(X, **SEG)
    Y = make_segments(Y, **SEG)
    if LAST:
        Y = Y[:, -1, :]

    logging.info(X.shape)
    logging.info(Y.shape)

    X_valid[name] = X
    Y_valid[name] = Y

# make test dataset
logging.info("-----test-----")

X_test = dict()
Y_test = dict()

for name in testlist:
    logging.info(name)

    X, Y = feature(name)
    X = normalize(X)

    logging.info(X.shape)
    logging.info(Y.shape)

    X_test[name] = X
    Y_test[name] = Y


# save
np.save(f"features/{DIRNAME}/X_train", X_train)
np.save(f"features/{DIRNAME}/Y_train", Y_train)

np.save(f"features/{DIRNAME}/X_valid", X_valid)
np.save(f"features/{DIRNAME}/Y_valid", Y_valid)

np.save(f"features/{DIRNAME}/X_test", X_test)
np.save(f"features/{DIRNAME}/Y_test", Y_test)

logging.info("Done")
