import os
import datetime
import logging
import argparse

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf


def peak(activation, pre_max, post_max, pre_avg, post_avg, delta, wait):
    peak = np.zeros(activation.shape, dtype="int32")

    peak_indices = librosa.util.peak_pick(
        activation, pre_max, post_max, pre_avg, post_avg, delta, wait
    ).astype(np.int32)

    peak[peak_indices] = 1

    return peak


def peaks(Y_activations, **kwargs):
    Y_peak_HH = peak(Y_activations[:, 0], **kwargs)
    Y_peak_SD = peak(Y_activations[:, 1], **kwargs)
    Y_peak_KD = peak(Y_activations[:, 2], **kwargs)

    Y_peak = np.hstack(
        (Y_peak_HH[:, np.newaxis], Y_peak_SD[:, np.newaxis], Y_peak_KD[:, np.newaxis])
    )

    return Y_peak


def accuracy(Y_annotation, Y_pred, pre_tolerance, post_tolerance):
    """
    Variables
    ----------
    Y_pred : ndarray (N,)
        1 if onset (pred) else 0
    Y_annotation : ndarray (N,)
        1 if onset (GT) else 0

    pre_tolerance : int
    post_tolerance : int

    Y_annotation[i] == 1 (real onset)
    Y_annotation[i] == 0 (real onset)
    Y_pred[i] == 1 (predicted onset)
    Y_pred[i] == 0 (predicted onset)

    """

    tp = 0
    fn = 0
    fp = 0

    N = len(Y_pred)
    for i in range(N):

        if Y_annotation[i] == 1:
            if i - pre_tolerance < 0:
                if 1 not in Y_pred[0 : i + post_tolerance]:
                    fn += 1
            elif N < i + post_tolerance:
                if 1 not in Y_pred[i - pre_tolerance : N]:
                    fn += 1
            elif 1 not in Y_pred[i - pre_tolerance : i + post_tolerance]:
                fn += 1

        # positive
        if Y_pred[i] == 1:
            if i - pre_tolerance < 0:
                if 1 in Y_annotation[0 : i + post_tolerance]:
                    tp += 1
                else:
                    fp += 1
            elif N < i + post_tolerance:
                if 1 in Y_annotation[i - pre_tolerance : N]:
                    tp += 1
                else:
                    fp += 1
            elif 1 in Y_annotation[i - pre_tolerance : i + post_tolerance]:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    fmeasure = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else np.nan

    if False:
        print("tp", tp)
        print("fp", fp)
        print("fn", fn)
        print("total pred", tp + fp)
        print("total annotation", tp + fn)
        print("precision", precision)
        print("recall", recall)
        print("fmeasure", fmeasure)

    return (
        tp,
        fp,
        fn,
        tp + fp,
        tp + fn,
        round(precision, 4),
        round(recall, 4),
        round(fmeasure, 4),
    )


def accuracies(Y_true, Y_peak, **kwargs):
    metrics_names = [
        "TP",
        "FP",
        "FN",
        "TP+FP",
        "TP+FN",
        "Precision",
        "Recall",
        "F-measure",
    ]
    inst_names = ["HH", "SD", "KD"]

    metrics_HH = accuracy(Y_true[:, 0], Y_peak[:, 0], **kwargs)
    metrics_SD = accuracy(Y_true[:, 1], Y_peak[:, 1], **kwargs)
    metrics_KD = accuracy(Y_true[:, 2], Y_peak[:, 2], **kwargs)

    metrics = metrics_HH + metrics_SD + metrics_KD

    n_metrics = len(metrics_names)
    n_inst = len(inst_names)

    index1 = ["HH"] * n_metrics + ["SD"] * n_metrics + ["KD"] * n_metrics
    index2 = metrics_names * n_inst

    return pd.Series(metrics, index=[index1, index2])


def eval_at(thres, name):
    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]

    # pred
    Y_pred = model(X_test[np.newaxis, :, :])[0].numpy()

    # peak_picking
    Y_peak = peaks(Y_pred, delta=thres, **PEAK)

    # eval
    result = accuracies(Y_test, Y_peak, **TOLERANCE)
    result.name = (thres, name)
    return result


def eval_all(thres_list, test_names):
    def eval_names_at(thres):
        series = [eval_at(thres, name) for name in test_names]
        result = pd.concat(series, axis=1).T
        mean = result.describe().loc["mean"]
        mean.name = (thres, "mean")
        mean = pd.DataFrame(mean).T
        return pd.concat((result, mean), axis=0)

    dfs = [eval_names_at(thres) for thres in thres_list]
    return pd.concat(dfs, axis=0)


if __name__ == "__main__":
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("feature")
    parser.add_argument("model")
    parser.add_argument("epoch", type=int)
    args = parser.parse_args()

    # params
    FEATURE = args.feature
    MODEL = args.model
    FOLD = -1
    EPOCH = int(args.epoch)
    PEAK = {"pre_max": 3, "post_max": 3, "pre_avg": 3, "post_avg": 3, "wait": 2}
    TOLERANCE = {"pre_tolerance": 3, "post_tolerance": 3}
    FIG = False
    DIRNAME = f"{FEATURE}_{MODEL}"

    now = datetime.datetime.now().strftime("%m%d-%H%M%S")
    os.makedirs(f"./results/{DIRNAME}/fig")

    # set logging
    logging.basicConfig(
        filename=f"./results/{DIRNAME}/eval_{now}.log", level=logging.INFO
    )
    logging.info("-----params")
    items = list(globals().items())
    for (symbol, value) in items:
        if symbol.isupper():
            logging.info(f"---{symbol}")
            logging.info(value)

    # load model
    cp_path = f"logs/{DIRNAME}/{FOLD}/cp-{EPOCH:04d}"
    model = tf.keras.models.load_model(cp_path)

    # load test data
    X_test_dict = np.load(f"features/{FEATURE}/X_test.npy", allow_pickle=True)[()]
    Y_test_dict = np.load(f"features/{FEATURE}/Y_test.npy", allow_pickle=True)[()]

    test_names = list(X_test_dict.keys())
    thres_list = [round(0.05 * i, 2) for i in range(1, 20)]

    # eval
    result = eval_all(thres_list, test_names)

    # save
    result.to_csv(f"results/{DIRNAME}/{FEATURE}_{MODEL}.csv")
