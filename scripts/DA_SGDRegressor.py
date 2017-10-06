from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from itertools import product
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn import linear_model, decomposition, datasets
print(__doc__)
from sklearn.linear_model import SGDRegressor
from sklearn import svm
def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
#
param_grid1 = {
#   'alpha' : [0.000001, 0.00001, 0.0001,0.001,0.01,0.1]
}


def main2(**kwargs):


    best_roc, best_args = 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    for args in iter_param_grid(param_grid):  # tried 5,6,7,8,9

        classifier = make_pipeline(
            MinMaxScaler(),
            svm.SVC(**args).fit(X_train, y_train)

        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        rs = roc_auc_score(y_test, y_pred)
        print(rs, args)
        if (rs > best_roc):
            best_roc, best_args = rs, args
    print("=== BEST PARAMETERS ===")
    print(best_roc, best_args)
    return classifier

def iter_param_grid(p):

    # Always sort the keys of a dictionary, for reproducibility
    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params






        #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main2(print_info=False)
    classifier = main2(print_info=False)
    df_train_all, df_hold = run(ds1='train_all', ds2='hold')
    X_train, y_train, X_test, y_test = extract(df_train_all, df_hold )
    y_pred = classifier.predict(X_test)
    rs = roc_auc_score(y_test, y_pred)
    print(rs)