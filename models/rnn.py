import tensorflow as tf


def train(X_train, X_valid, Y_train, Y_valid, params):

    # model
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(params["n_hidden"], activation=params["activation"]),
        tf.keras.layers.Dense(params["n_out"], activation='sigmoid')
    ])

    model.compile(optimizer=params["optimizer"],
                  loss=params["loss"],
                  metrics=[params["metrics"]])

    # set callback
    tensorboard_callback = ""
    cp_callback = ""

    # train
    model.fit(X_train,
              Y_train,
              batch_size=params["batch_size"],
              epochs=params["epochs"],
              #callbacks=[tensorboard_callback, cp_callback],
              validation_data=(X_valid, Y_valid))
