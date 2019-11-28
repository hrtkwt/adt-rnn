from sklearn.model_selection import KFold
from utils.data import load_datasets, load_target
from models.rnn import train


# load parameter

X_train_all, X_test = load_datasets(feats="")
Y_train_all = load_target(target_name="")

rnn_params = ""

kf = KFold(n_splits=3, random_state=0)

for train_index, valid_index in kf.split(X_train_all):

    X_train, X_valid = X_train_all[train_index, :], X_train_all[valid_index, :]
    Y_train, Y_valid = Y_train_all[train_index], Y_train_all[valid_index]

    train(X_train, X_valid, Y_train, Y_valid, rnn_params)
