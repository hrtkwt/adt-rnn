import os
import xml.etree.ElementTree as ET

import numpy as np
import librosa
import librosa.display


def get_func_feature(mode):
    if mode == "a":
        return feature_a
    if mode == "smo5":
        return feature_smooth_5


def feature_a(name):
    y = load_audio(name, inst="#MIX")
    specs = get_specs(y)

    targets = get_targets(name).T

    X = specs["a"]
    Y = targets

    return X, Y


def feature_smooth_5(name):
    y = load_audio(name, inst="#MIX")
    specs = get_specs(y)

    smo5 = smooth_spec_5(specs["pdd2"])

    targets = get_targets(name).T

    X = np.hstack((specs["a"], smo5))
    Y = targets

    return X, Y


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


def get_targetpath(audioname, inst):
    targetpath = "/home/cs181004/data/SMT_DRUMS/annotation_XX/"
    targetpath = os.path.join(targetpath, audioname + inst + ".svl")

    return targetpath


def get_audiolen(audioname):
    y = load_audio(audioname, inst="#MIX")
    return len(y)


def load_audio(audioname, inst):
    audiopath = get_audiopath(audioname, inst)
    y, _ = librosa.load(audiopath, sr=44100)
    return y


def load_target(audioname, inst):
    targetpath = get_targetpath(audioname, inst)
    length = get_audiolen(audioname)

    tree = ET.parse(targetpath)

    root = tree.getroot()
    dataset = root[0][1]

    # sampling rate 44100 における オンセットのフレームを取得
    onsets = []
    for data in dataset:
        # data.tag -> point
        # data.attrib -> {'frame': '759638', 'label': 'New Point'}

        onset = data.attrib["frame"]
        onsets.append(onset)
    onsets = np.array(onsets, dtype=np.int32)

    # audio のlength 分で初期化
    target = np.zeros(length, dtype=np.float32)
    # onsetのフレームで初期化
    target[onsets] = 1

    return target


def down_target(target):
    # onsets(sr=44100におけるフレーム番号）を取得
    # hop_length -> 512 と固定
    target_length = len(target)

    onsets = np.where(target == 1)[0]

    # plus 1 する理由: stftで center=true にすると行き過ぎたフレームを取り込む場合がある
    target_length_down = (target_length // 512) + 1

    hop_points = np.array([i * 512 for i in range(target_length)])

    onsets_down = []
    for onset in onsets:
        # hop_pointsの内 onsetフレームに最も近い indexを
        onset_down = np.abs(hop_points - onset).argmin()
        onsets_down.append(onset_down)
    onsets_down = np.array(onsets_down)

    target_down = np.zeros(target_length_down)
    target_down[onsets_down] = 1
    return target_down


def get_specs(y):
    C = librosa.core.stft(
        y,
        n_fft=2048,
        hop_length=2048 // 4,
        window="hann",
        center=True,
        dtype=np.complex64,
        pad_mode="reflect",
    ).T

    C_a = np.abs(C)
    C_p = np.angle(C)
    C_up = np.unwrap(C_p, axis=0)

    C_pd = np.zeros(C_up.shape)
    for j in range(C_up.shape[1]):
        for i in range(C_up.shape[0] - 1):
            C_pd[i, j] = C_up[i + 1, j] - C_up[i, j]

    C_pdd = np.zeros(C_pd.shape)
    for j in range(C_up.shape[1]):
        for i in range(C_up.shape[0] - 1):
            C_pdd[i, j] = C_pd[i + 1, j] - C_pd[i, j]

    C_pdd2 = apply_maxmin(-np.abs(C_pdd))

    return {"a": C_a, "p": C_p, "up": C_up, "pd": C_pd, "pdd": C_pdd, "pdd2": C_pdd2}


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

    segments = np.array(segments, dtype=np.float32)

    return segments


def get_targets(audioname):
    targets = []
    for inst in ["#HH", "#SD", "#KD"]:
        target = load_target(audioname, inst)
        target = down_target(target)
        targets.append(target)

    return np.array(targets)


def apply_zscore(x):
    return (x - x.mean()) / x.std()


def apply_maxmin(x):
    return (x - x.min()) / (x.max() - x.min())


def smooth_spec(C, start, end):
    return np.sum(C[:, start:end], axis=1)


def smooth_specs(**kwargs):
    out = {}
    for name, C in kwargs.items():
        C = smooth_spec(C)
        out[name] = C

    return out


def smooth_spec_5(C):
    step = 205

    i = 0
    start = i * step
    end = (i + 1) * step

    wave = smooth_spec(C, start, end)
    wave = pdd3(wave)[:, np.newaxis]

    out = wave

    for i in range(1, 5):
        start = i * step
        end = (i + 1) * step

        wave = smooth_spec(C, start, end)[:, np.newaxis]
        wave = pdd3(wave)

        out = np.hstack((out, wave))

    return out


def pdd3(arr):
    arr = prepro(arr)
    arr = np.abs(arr - arr.mean())
    arr = apply_maxmin(arr)

    return arr


def prepro(y):
    m = np.mean(y)
    y[0] = m
    y[1] = m
    y[-1] = m
    y[-2] = m
    return y


def load_pesudo_dataset():
    X_train = np.random.randn(1000, 100, 1025)
    Y_train = np.random.randn(1000, 3)

    X_test = np.random.randn(200, 1025)
    Y_test = np.random.randn(200, 3)

    return X_train, X_test, Y_train, Y_test
