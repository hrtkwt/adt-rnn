import tensorflow as tf


def select_model(mode):
    if mode == "base_1025":
        return create_base_1025()
    if mode == "base_2050":
        return create_base_2050()
    else:
        exit(1)


def create_base_1025():
    # model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(
                units=400,
                activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                return_sequences=True,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                input_shape=(None, 1025)
            ),
            tf.keras.layers.Dense(
                units=3,
                activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            ),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',)

    loss = tf.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None,
    )
#    model.summary()
    return model


def create_base_2050():
    # model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(
                units=400,
                activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                return_sequences=True,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                input_shape=(None, 2050)
            ),
            tf.keras.layers.Dense(
                units=3,
                activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            ),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',)

    loss = tf.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None,
    )
#    model.summary()
    return model
