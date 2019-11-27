from sklearn.model_selection import KFold
from utils import load_datasets, load_target

# load parameter

X_train_all, X_test = load_datasets(feats="")
y_train_all = load_target(target_name="")

y_preds = []
models = []

kf = KFold(n_splits=3, random_state=0)
for train_index, valid_index in kf.split(X_train_all):
    X_train, X_valid = X_train_all[train_index, :], X_train_all[valid_index, :]
    y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]
