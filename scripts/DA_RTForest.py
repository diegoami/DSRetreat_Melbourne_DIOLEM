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
def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test



def main(**kwargs):


    best_roc, best_params = 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    for max_features in ['auto', 'sqrt', 'log2']:  # tried 5,6,7,8,9
        for min_samples_split in [1, 2, 3]:  # tried 5,6,7,8
            for n_estimators in [10, 50, 100]:  # tried 5,6,7,8

                classifier = make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(max_features=max_features, min_samples_split=min_samples_split, n_estimators=n_estimators)
                )
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                rs = roc_auc_score(y_test, y_pred)
                print(rs, max_features, min_samples_split)
                if (rs > best_roc):
                    best_roc, best_params = rs, (
                        max_features, min_samples_split)
    print(best_roc, best_params)


param_grid1 = {
  # 'n_estimators': [200, 400, 800, 1500],
    'min_samples_leaf': [5,  15, 25, 30, 50],
    'min_samples_split': [ 2,  4, 6, 10],
    'max_features' : [5,7,11,20],
    'n_jobs' : [-1]
}

param_grid2 = {
    'n_estimators': [20, 200],
    'min_samples_leaf': [20, 23, 25,28],
    'min_samples_split': [ 3, 4, 5],
    'max_features' : [6,7,8,9,10],
    'n_jobs' : [-1]
}

param_grid3 = {
    'n_estimators': [ 200, 400, 600],
    'min_samples_leaf': [26, 28, 30],
    'min_samples_split': [  4, 5, 6],
    'max_features' : [7,8,9],
    'n_jobs' : [-1]
}

param_grid6 = {
    'n_estimators': [ 600, 800, 1000],
    'min_samples_leaf': [25, 26, 27, 30],
    'min_samples_split': [ 3, 4, 5],
    'max_features' : [8,11],
    'n_jobs' : [-1]
}
# 0.72616648003 {'max_features': 8, 'min_samples_leaf': 25, 'min_samples_split': 5, 'n_estimators': 600, 'n_jobs': -1}

def main2(**kwargs):


    best_roc, best_args = 0, None
    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)
    for args in iter_param_grid(param_grid6):  # tried 5,6,7,8,9

        classifier = make_pipeline(
            MinMaxScaler(),
            decomposition.PCA(),
            RandomForestRegressor(**args)
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

# NO PCA 0.681083026716 {'max_features': 8, 'min_samples_leaf': 20, 'min_samples_split': 3, 'n_estimators': 200, 'n_jobs': -1}


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

   classifier = main2(print_info=False)
   df_train_all, df_hold = run(ds1='train_all', ds2='hold')
   X_train, y_train, X_test, y_test = extract(df_train_all, df_hold )
   y_pred = classifier.predict(X_test)
   rs = roc_auc_score(y_test, y_pred)
   print(rs)