import numpy as np
import pandas as pd
import os.path
from scripts.DA_join_apps_rfcd_seos import main

def fix():
    pass

def set_categories(data, cat_list):
    for l in cat_list:
        data[l] = data[l].astype('category')
        data[l] = data[l].cat.codes
    return data


def treat_missing_columns(train,test):
    relevant_columns = [x for x in set(list(train.columns) + list(test.columns))
                        if x not in ['id', 'granted']]

    # get relevant columns
    missing_train_columns = [x for x in relevant_columns if x not in train]
    missing_test_columns = [x for x in relevant_columns if x not in test]

    train.reindex(columns=list(train.columns) + missing_train_columns)
    test = test.reindex(columns=(list(test.columns) + missing_test_columns))

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    return train, test


def run(ds1='train',ds2='test'):
    cat_list = ['sponsor', 'grant_category', 'Dept.No.', 'Faculty.No.','month']
    base = '../data/'
    trainf = ds1+ '_ml.csv'
    testf =  ds2+ '_ml.csv'

    train = pd.read_csv(os.path.join(base, trainf), low_memory=False, parse_dates=['date'])
    test = pd.read_csv(os.path.join(base, testf), low_memory=False, parse_dates=['date'])
    print(train.head(),test.head())
    train = set_categories(train,cat_list)
    test = set_categories(test, cat_list)

    train, test = treat_missing_columns(train, test)

    train = train.drop('date', axis=1)
    test = test.drop('date', axis=1)
    print(train.head(),test.head())
    return train, test


if __name__ == '__main__':
    run()