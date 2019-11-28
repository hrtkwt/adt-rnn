import numpy as np


def load_train_data(date):
    if date == 'pesu':
        rand = np.random.randn(1000, 100, 1025)
        return rand


def load_test_data(date):
    if date == 'pesu':
        rand = np.random.randn(200, 100, 1025)
        return rand


def load_train_target(date):
    if date == 'pesu':
        rand = np.random.randn(1000, 3)
        return rand

def load_test_target(date):
    if date == 'pesu':
        rand = np.random.randn(200, 3)
        return rand


if __name__ == '__main__':
    X_train = load_train_data(date="pesu")
    X_test = load_test_data(date="pesu")
    Y_train = load_train_target(date="pesu")
    Y_test = load_test_target(date="pesu")

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
