import tensorflow as tf
import numpy as np


def create_model(params):
    # model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(
                params["n_hidden"], activation=params["activation"]
            ),
            tf.keras.layers.Dense(params["n_out"], activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=params["optimizer"], loss=params["loss"], metrics=[params["metrics"]]
    )

    return model


def train(X_train, X_valid, Y_train, Y_valid, params, now, fold):

    # model
    model = create_model(params)

    # set callbacks
    # tb_callback
    logdir = f"logs/{now}/{fold}"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # cp_callback
    cp_path = f"cp/{now}/{fold}" + "/cp-{epoch:04d}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_path, save_weights_only=False, verbose=1
    )

    # train
    model.fit(
        X_train,
        Y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        callbacks=[tb_callback, cp_callback],
        validation_data=(X_valid, Y_valid),
    )

    return model


def get_weights(train_date, fold, epoch):
    def load_model(train_date, fold, epoch):
        cp_path = f"cp/{train_date}/{fold}/cp-{epoch:04d}"
        model = tf.keras.models.load_model(cp_path)
        return model

    model = load_model(train_date, fold, epoch)
    weights = [w.numpy() for w in model.weights]
    return weights


def pred(X, Wh, Wr, bh, Wo, bo):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def feed_forward(x, h_):
        zh = np.dot(x, Wh) + np.dot(h_, Wr) + bh
        h = sigmoid(zh)

        zo = np.dot(h, Wo) + bo
        y = sigmoid(zo)

        return y, h

    x0 = X[0]
    h0 = np.zeros(200)

    Y, h = feed_forward(x0, h0)

    for x in X[1:]:
        y, h = feed_forward(x, h)
        Y = np.vstack((Y, y))

    return Y
