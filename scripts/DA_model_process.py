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
sys.path.append('..')
import pandas as pd

def extract(df_train, df_test):

    y_train = df_train['granted']
    y_test = df_test['granted']
    X_train = df_train.drop(['granted', 'id'],axis=1)
    X_test = df_test.drop(['granted', 'id'],axis=1)

    return X_train, y_train, X_test, y_test

def get_random_classifier():
    return RandomForestClassifier(n_estimators=200)


def get_random_regressor():
    return RandomForestRegressor(n_estimators=200)



def get_classifier_xgboost():
    return XGBClassifier(max_depth=2, min_child_weight=2, gamma=0),


def get_best_xgboost():
    return XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.8, colsample_bytree=0.7, reg_alpha = 0.5,learning_rate =0.1, n_estimators= 500)

def get_gridsearch_xgboost():
    xgbparams = {
        #'min_child_weight' : [1,2,3,4,5],
        #'max_depth' : [1,2,3,4,5]
        #'gamma': [0.38, 0.39, 0.4, 0.41, 0.42]
        #'subsample': [0.75,  0.775, 0.8, 0.825, 0.85],
        #'colsample_bytree': [0.65, 0.675, 0.7, 0.725, 0.75]
        #'reg_alpha': [0.1, 0.5 , 1, 5, 10]
        #'learning_rate' : [0.1 , 0.001, 0.0001],
        #'n_estimators': [400, 1500, 5000],
 }
    gs = GridSearchCV(XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.8, colsample_bytree=0.7, reg_alpha = 0.5, learning_rate =0.1, n_estimators= 500  ), xgbparams,cv =5, scoring='roc_auc', verbose = 10)
    return gs


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

def main(**kwargs):

    df_train, df_test = run(**kwargs)
    xgridboost = get_gridsearch_xgboost()
    test_with_classif(xgridboost , df_train, df_test)
    bestboost = get_best_xgboost()

    test_with_classif(bestboost , df_train, df_test)
    series = pd.Series(bestboost.get_booster().get_fscore())
    print(series.sort_values(ascending=False))


if __name__ == "__main__":

    main()