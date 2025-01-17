import numpy as np
import tensorflow as tf
import librosa
import pandas as pd


def get_weights(cp_path):
    def load_model(cp_path):
        model = tf.keras.models.load_model(cp_path)
        return model

    model = load_model(cp_path)
    weights = [w.numpy() for w in model.weights]
    return weights


def peak_picking(activation, pre_max, post_max, pre_avg, post_avg, delta, wait):
    peak = np.zeros(activation.shape, dtype="int32")

    peak_indices = librosa.util.peak_pick(
        activation, pre_max, post_max, pre_avg, post_avg, delta, wait
    ).astype(np.int32)

    peak[peak_indices] = 1

    return peak


def peaks(Y_activations, **kwargs):
    """
    param
    pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.05, wait=2
    """

    Y_peak_HH = peak_picking(Y_activations[:, 0], **kwargs)
    Y_peak_SD = peak_picking(Y_activations[:, 1], **kwargs)
    Y_peak_KD = peak_picking(Y_activations[:, 2], **kwargs)

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


metrics_names = ["TP", "FP", "FN", "TP+FP", "TP+FN", "Precision", "Recall", "F-measure"]
inst_names = ["HH", "SD", "KD"]


def accuracies(Y_true, Y_peak, **kwargs):
    metrics_HH = accuracy(Y_true[:, 0], Y_peak[:, 0], **kwargs)
    metrics_SD = accuracy(Y_true[:, 1], Y_peak[:, 1], **kwargs)
    metrics_KD = accuracy(Y_true[:, 2], Y_peak[:, 2], **kwargs)

    metrics = metrics_HH + metrics_SD + metrics_KD

    n_metrics = len(metrics_names)
    n_inst = len(inst_names)

    index1 = ["HH"] * n_metrics + ["SD"] * n_metrics + ["KD"] * n_metrics
    index2 = metrics_names * n_inst

    return pd.Series(metrics, index=[index1, index2])


def get_result_table(audio_names):
    n_audio = len(audio_names)
    n_metrics = len(metrics_names)
    n_inst = len(inst_names)

    column1 = ["HH"] * n_metrics + ["SD"] * n_metrics + ["KD"] * n_metrics
    column2 = metrics_names * n_inst

    result_table = pd.DataFrame(
        np.zeros(n_audio * n_metrics * n_inst).reshape(n_audio, n_metrics * n_inst),
        index=audio_names,
        columns=[column1, column2],
    )
    result_table.index.names = ["audio"]
    result_table.columns.names = ["inst", "metrics"]

    return result_table


def info(C):
    print("    shape:", C.shape)
    print("    max:", np.max(C))
    print("    min:", np.min(C))
    print("    median:", np.median(C))
    print("    mean:", np.mean(C))
    print("    std:", np.std(C))


def get_stats(C):
    stats = {
        "size": C.size,
        "max": np.max(C),
        "min": np.min(C),
        "median": np.median(C),
        "mean": np.mean(C),
        "std:": np.std(C),
    }
    return stats
