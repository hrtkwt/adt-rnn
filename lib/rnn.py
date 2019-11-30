import tensorflow as tf
from tensorflow import keras


def create_model(params):
    # model
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(params["n_hidden"], activation=params["activation"]),
        tf.keras.layers.Dense(params["n_out"], activation='sigmoid')
        ])

    model.compile(
        optimizer=params["optimizer"],
        loss=params["loss"],
        metrics=[params["metrics"]]
        )

    return model

def load_model(now, fold, epoch):
    cp_path = f"cp/{now}/{fold}/cp-{epoch:04d}"
    model = tf.keras.models.load_model(cp_path)
    return model

def train(X_train, X_valid, Y_train, Y_valid, params, now, fold):

    # model
    model = create_model(params)

    # set callbacks
    ## tb_callback
    logdir = f"logs/{now}/{fold}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    ## cp_callback
    cp_path = f"cp/{now}/{fold}" + "/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_path,
        save_weights_only=True,
        verbose=1
        )

    # train
    model.fit(
        X_train,
        Y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        callbacks=[tb_callback, cp_callback],
        validation_data=(X_valid, Y_valid)
        )
