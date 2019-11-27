import numpy as np


def load_datasets(feats):
    if feats == 'pesu':
        rand_train = np.random.randn(1000, 100, 1025)
        rand_test = np.random.randn(200, 100, 1025)
        return rand_train, rand_test


def load_target(target_name):
    if target_name == 'pesu':
        rand = np.random.randn(1000)
        return rand


if __name__ == '__main__':
    X_train_all, X_test_all = load_datasets(feats="pesu")
    y_train_all = load_target(target_name="pesu")

    print(X_train_all.shape)
    print(X_test_all.shape)
    print(y_train_all.shape)
