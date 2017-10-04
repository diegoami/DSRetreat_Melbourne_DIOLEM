#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os.path


def create_apps(df):
    apps = df.iloc[:, :6]
    replace_grant_values(apps)
    return apps


def replace_grant_values(apps):
    rep = {'A': 5000,
           'B': 10000,
           'C': 20000,
           'D': 30000,
           'E': 40000,
           'F': 50000,
           'G': 100000,
           'H': 200000,
           'I': 300000,
           'J': 400000,
           'K': 500000,
           'L': 600000,
           'M': 700000,
           'N': 800000,
           'O': 900000,
           'P': 1000000,
           'Q': 10000000}
    apps['grant_value'] = apps['grant_value'].str.strip()
    apps['grant_value'].replace(rep, inplace=True)


def process(dataset = 'train',):
    base = '../data/'
    readfile = dataset+'.csv'
    writefile = dataset + '_apps_raw.csv'

    df = pd.read_csv(os.path.join(base, readfile), low_memory=False)
    apps = create_apps(df)
    apps.to_csv(os.path.join(base, writefile), index=False)


if __name__ == '__main__':
    process()