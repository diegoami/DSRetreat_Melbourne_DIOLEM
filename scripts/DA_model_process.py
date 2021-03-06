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
from matplotlib import pyplot as plt

def do_lasso(df_train, df_test):
    def get_best_lasso():
        return Lasso(alpha=0.0005,max_iter=10000)

    def get_gridsearch_lasso():
        params = {
            #"alpha" : [0.0001, 0.0005,0.00005]
          }
        gs = GridSearchCV(
            Lasso() , params, cv=5, scoring='roc_auc', verbose=10)
        return gs

   # lasso = get_gridsearch_lasso()
   # test_with_classif(lasso, df_train, df_test)

    bestboost = get_best_lasso()

    test_with_classif(bestboost , df_train, df_test)
    return bestboost


def do_rtree(df_train, df_test):
    def get_best_rtree():
        return RandomForestRegressor(min_samples_leaf=1,min_samples_split=26, n_estimators= 50)

    def get_gridsearch_rtree():
        rtparams = {
            #"min_samples_leaf" : (1,2,3)
            #"min_samples_split" : (20,22,24,26,28)
            #"n_estimators" : (10,50,100,500,1000)
        }
        rt = GridSearchCV(
            RandomForestRegressor(min_samples_leaf=1,min_samples_split=26, n_estimators= 50) , rtparams, cv=5, scoring='roc_auc', verbose=10)
        return rt

    #rtree = get_gridsearch_rtree()
    #test_with_classif(rtree, df_train, df_test)
    bestboost = get_best_rtree()

    test_with_classif(bestboost , df_train, df_test)
    return bestboost

def do_xgboost(df_train, df_test):
    def get_best_xgboost1():
        return XGBRegressor(min_child_weight=3, max_depth=4,  gamma=0.1, subsample=1, colsample_bytree=0.7,
                            reg_alpha=0.5, learning_rate=0.1, n_estimators=2000)

    def get_best_xgboost2():
        return XGBRegressor(min_child_weight=7, max_depth=6,
                            gamma=0.4, subsample=0.7,
                            colsample_bytree = 0.7, learning_rate=0.1, reg_alpha=0.9, n_estimators=2000)

    def get_best_xgboost3():
        return XGBRegressor(max_depth=2, min_child_weight=2, gamma=0.8, subsample=0.4, colsample_bytree=0.7,
                            learning_rate= 0.0095, reg_alpha=0.3, n_estimators=2000,scale_pos_weight=1)

    def get_best_xgboost4():
        return XGBRegressor(max_depth=2, min_child_weight=3, gamma=0.6, subsample=0.4, colsample_bytree=0.7,
                            learning_rate=0.0095, reg_alpha=0.3, n_estimators=2000, scale_pos_weight=1)

    def get_gridsearch_xgboost():
        xgbparams = {
             #'min_child_weight' : [7,8,9],
             #'max_depth' : [5,6,7]
             #'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
             #'subsample': [0.5,  0.6, 0.7, 0.8, 0.9, 1],
             #'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9]
             'reg_alpha': [0.1, 0.01 , 1, 10, 100]
             #'learning_rate' : [0.1 , 0.001, 0.0001],
             #'n_estimators': [400, 1500, 5000],
        }
        gs = GridSearchCV(
            XGBRegressor(max_depth=6, min_child_weight=7, gamma=0.4, subsample=0.7, colsample_bytree = 0.5), xgbparams, cv=5, scoring='roc_auc', verbose=10)
        return gs


    #xgridboost = get_gridsearch_xgboost()
    #test_with_classif(xgridboost , df_train, df_test)
    for bestboost in [
        get_best_xgboost1(),
        get_best_xgboost2(),
        get_best_xgboost3(), get_best_xgboost4()]:
        test_with_classif(bestboost, df_train, df_test)
        series = pd.Series(bestboost.get_booster().get_fscore())
        print(series.sort_values(ascending=False))
        series.plot(kind='bar', title='Feature Importances')

    #test_with_classif(XGBRegressor(),xgbparams , df_train, df_test)
def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)

    return X_train, y_train, X_test, y_test


def print_best_parameters(classifier):
    if hasattr(classifier, "best_estimator_"):
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))




def test_with_classif(classifier, df_train, df_test):

    X_train, y_train, X_test, y_test  = extract(df_train, df_test)
    classifier.fit(X=X_train, y=y_train)

    y_pred = classifier.predict(X_test)
    print_best_parameters(classifier)
    print(roc_auc_score(y_test, y_pred))
    if (hasattr(classifier, "_coef")):
        coef = pd.Series(classifier.coef_, index=X_train.columns)
        print(coef)


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
    classifiers = [
        #do_lasso(df_train, df_test),
        do_xgboost(df_train, df_test),
       # do_rtree(df_train, df_test)
    ]

    #test_with_classifs(classifiers, df_train, df_test)
if __name__ == "__main__":
    print(" ===== RESULTS ON TEST ===========")
    main(ds1='train',ds2='test',print_info=False)
    print(" ===== RESULTS ON HOLD ===========")

    main(ds1='train_all', ds2='hold', print_info=False)


