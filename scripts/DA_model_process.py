from sklearn.ensemble import  RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import  RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import sys

sys.path.append('..')
import pandas as pd


def extract(df_train, df_test):
    cat_columns = ['sponsor', 'grant_category']
    for cat_column in cat_columns:
        df_train[cat_column] = df_train[cat_column].astype('category')
        df_train[cat_column] = df_train[cat_column].cat.codes
        df_test[cat_column] = df_test[cat_column].astype('category')
        df_test[cat_column] = df_test[cat_column].cat.codes


    df_train = df_train.drop('date', axis=1)
    df_test = df_test.drop('date', axis=1)

    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)

    relevant_columns = [x for x in set(list(df_train.columns)+list(df_test.columns)) if x not in ['id', 'granted']]
    #print(relevant_columns)
    missing_train_columns = [x for x in relevant_columns if x not in df_train ]


    missing_test_columns = [x for x in relevant_columns if x not in df_test ]
    print(missing_test_columns)

    df_train.reindex(columns=list(df_train.columns)+missing_train_columns)

    df_test = df_test.reindex(columns=(list(df_test.columns) + missing_test_columns))

    X_train = df_train[relevant_columns]
    y_train = df_train['granted']
    X_test = df_test[relevant_columns]
    y_test = df_test['granted']

    y_train.fillna(0, inplace=True)
    y_test.fillna(0, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_train, y_train, X_test, y_test

def get_random_classifier():
    return RandomForestClassifier(n_estimators=200)

def get_random_regressor():
    return RandomForestRegressor(n_estimators=200)


def get_regressor_xgboost():
    return XGBRegressor(max_depth=3, min_child_weight=1)


def get_regressor_xgboost2():
    return XGBRegressor(max_depth=5, min_child_weight=3)


def print_best_parameters(classifier):
    if hasattr(classifier, "best_estimator_"):
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


def test_with_classif(classifier, df_train, df_test):
    print(classifier)
    X_train, y_train, X_test, y_test  = extract(df_train, df_test)

    classifier.fit(X=X_train, y=y_train)
    y_pred = classifier.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    print_best_parameters(classifier)
    print(roc_auc_score(y_test, y_pred))
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(scores)

if __name__ == "__main__":

    df_train = pd.read_csv('../data/train_ml.csv', parse_dates=True)
    df_test = pd.read_csv('../data/test_ml.csv', parse_dates=True)

    test_with_classif(get_regressor_xgboost2(), df_train, df_test)

    test_with_classif(get_regressor_xgboost(),df_train, df_test)
    test_with_classif(get_random_regressor(),df_train, df_test)