import tensorflow as tf


def train(X_train, X_valid, Y_train, Y_valid, params):

    BATCH_SIZE = 10
    EPOCHS = 20

    # model
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(200, activation='sigmoid'),
        tf.keras.layers.Dense(3, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    tensorboard_callback = ""
    cp_callback = ""

    # train
    model.fit(X_train,
              Y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[tensorboard_callback, cp_callback],
              validation_split=0.2)


if __name__ == '__main__':
    print("model")
