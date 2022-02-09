# Random Forest Classifier

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from joblib import Memory, Parallel, delayed
location = './cachedir'
memory = Memory(location, verbose=0)

DATASET_FILENAME = '../datasets/diabetes_complete.csv' # change filename depending on dataset used

def get_score(pred, y):
    # loss = nn.BCELoss()
    # return loss(torch.tensor(pred).float(), torch.tensor(y).float()).item()
    return roc_auc_score(y, pred)


@memory.cache
def run_one_iter(it, n_features):


    # load data from dataset
    data = np.genfromtxt(DATASET_FILENAME, delimiter=',')
    data = data[1:]  # remove header

    X = []
    y = np.empty(len(data))
    for i in range(len(data)):
        y[i] = data[i][-1:]
        X.append(np.delete(data[i], -1))  # remove last element

    X = np.asarray(X)
    y = np.asarray(y)

    X_test = X[0:n_test]
    y_test = y[0:n_test]
    X_val = X[n_test:(n_test + n_val)]
    y_val = y[n_test:(n_test + n_val)]
    X_train = X[(n_test + n_val):]
    y_train = y[(n_test + n_val):]

    # 5864 samples
    X_train, y_train = make_classification(n_samples=5846, n_features=4, n_informative=4, n_redundant=0)
    clf = RandomForestClassifier(max_depth=8, random_state=None)

    clf.fit(X_train, y_train)

    pred_val = clf.predict(X_val)
    pred_val = get_score(pred_val, y_val)

    print('AUROC = {}'.format(pred_val))


if __name__ == '__main__':
    n = int(614)
    n_test = int(77)
    n_val = int(77)
    n_iter = 1
    n_features = 4
    n_jobs = 1

    results = Parallel(n_jobs=n_jobs)(
            delayed(run_one_iter)(it, n_features=n_features)
            for it in range(n_iter)
        )