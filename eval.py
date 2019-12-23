import os
import datetime
import logging
import argparse
import io

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt


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


def get_prf(Y_annotation, Y_pred, pre_tolerance, post_tolerance):
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


def get_bce(Y_true, Y_pred):
    bce = tf.losses.BinaryCrossentropy()(Y_true, Y_pred)
    return round(bce.numpy(), 4)


def eval_at(thres, name):
    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]

    # pred
    Y_pred = model(X_test[np.newaxis, :, :])[0].numpy()

    # peak_picking
    Y_peak = peaks(Y_pred, delta=thres, **PEAK)

    metrics = get_prf(Y_test[:, 0], Y_peak[:, 0], **TOLERANCE)
    bce = get_bce(Y_test[:, 0], Y_pred[:, 0])
    HH = metrics + (bce,)

    metrics = get_prf(Y_test[:, 1], Y_peak[:, 1], **TOLERANCE)
    bce = get_bce(Y_test[:, 1], Y_pred[:, 1])
    SD = metrics + (bce,)

    metrics = get_prf(Y_test[:, 2], Y_peak[:, 2], **TOLERANCE)
    bce = get_bce(Y_test[:, 2], Y_pred[:, 2])
    KD = metrics + (bce,)

    index1 = ["HH"] * 9 + ["SD"] * 9 + ["KD"] * 9
    index2 = [
        "TP",
        "FP",
        "FN",
        "TP+FP",
        "TP+FN",
        "Precision",
        "Recall",
        "F-measure",
        "bce",
    ] * 3

    result = pd.Series(HH + SD + KD, index=[index1, index2])
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


def fig(name, i):
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def spec(X_test):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(10, 10))
        plt.title(name)
        plt.xlabel("time")
        plt.ylabel("freq")
        plt.grid(False)
        plt.imshow(X_test.T[::-1])
        plt.colorbar()

        return figure

    def activation(inst):
        if inst == "HH":
            i = 0
            color = "b"
        elif inst == "SD":
            i = 1
            color = "g"
        elif inst == "KD":
            i = 2
            color = "r"

        figure = plt.figure(figsize=(15, 5))
        plt.subplot(311)
        plt.xlabel("time")
        plt.ylim((0, 1.0))
        plt.title(name)
        plt.plot(Y_test[:, i], color=color)
        plt.subplot(312)
        plt.xlabel("time")
        plt.ylim((0, 1.0))
        plt.plot(Y_pred[:, i], color=color)
        plt.subplot(313)
        plt.xlabel("time")
        plt.ylim((0, 1.0))
        plt.plot(Y_peak[:, i], color=color)

        return figure

    X_test = X_test_dict[name]
    Y_test = Y_test_dict[name]
    Y_pred = model(X_test[np.newaxis, :, :])[0].numpy()
    Y_peak = peaks(Y_pred, delta=0.05, **PEAK)

    logdir = f"logs/{DIRNAME}/{-1}/test"
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    # Prepare the plot
    fig = spec(X_test)
    with file_writer.as_default():
        tf.summary.image("feature", plot_to_image(fig), i)

    fig = activation("HH")
    with file_writer.as_default():
        tf.summary.image("HH", plot_to_image(fig), i)

    fig = activation("SD")
    with file_writer.as_default():
        tf.summary.image("SD", plot_to_image(fig), i)

    fig = activation("KD")
    with file_writer.as_default():
        tf.summary.image("KD", plot_to_image(fig), i)


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
    os.makedirs(f"./results/{DIRNAME}/")

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

    for i, name in enumerate(test_names):
        fig(name, i)
