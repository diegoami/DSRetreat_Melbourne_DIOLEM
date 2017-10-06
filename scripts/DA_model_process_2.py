from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# from scripts.DA_model_extract import extract, extract_old
import sys
from scripts.eo_transport_data import run
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso


def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)
    return X_train, y_train, X_test, y_test



def test_with_classif_manual(classifier, parameters,df_train, df_test):

    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    #for 'min_child_weight' in [1, 2, 3, 4, 5]:
    #print([score(y_test, roc_auc_score(**args).fit(X_train, y_train).predict(X_test)) for args in parameters])

def test_with_classifs(classifiers, df_train, df_test):


    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    y_preds = [classifier.predict(X_test) for classifier in classifiers]
    y_pred_concat = np.vstack(y_preds)
    print(y_pred_concat.shape)
    y_pred_average = y_pred_concat.T.mean(axis=1)
    print(y_pred_average.shape, y_test.shape)
    print(roc_auc_score(y_test, y_pred_average))



def main(**kwargs):

    df_train, df_test = run(**kwargs)
    X_train, y_train, X_test, y_test = extract(df_train, df_test)

    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":

    main(print_info=False)