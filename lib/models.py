import tensorflow as tf
import numpy as np


def select_model(mode):
    if mode == "RNN1":
        return RNN1()


class RNN1(tf.keras.Model):
    def __init__(self):
        super(RNN1, self).__init__()
        self.hidden = tf.keras.layers.SimpleRNN(
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
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
        )
        self.out = tf.keras.layers.Dense(
            units=3,
            activation="sigmoid",
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )

    def call(self, x):
        x = self.hidden(x)
        return self.out(x)

    def unfold(self, X):
        def variables():
            Wh, Wr, bh = self.hidden.weights
            Wo, bo = self.out.weights

            return Wh, Wr, bh, Wo, bo

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def feed_forward(x, h_):
            zh = np.dot(x, Wh) + np.dot(h_, Wr) + bh
            h = sigmoid(zh)

            zo = np.dot(h, Wo) + bo
            y = sigmoid(zo)

            return y, h

        Wh, Wr, bh, Wo, bo = variables()

        x0 = X[0]
        h0 = np.zeros(bh.shape)

        Y, h = feed_forward(x0, h0)

        for x in X[1:]:
            y, h = feed_forward(x, h)
            Y = np.vstack((Y, y))

        return Y
