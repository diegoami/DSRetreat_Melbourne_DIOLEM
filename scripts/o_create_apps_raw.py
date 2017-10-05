#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os.path
import json

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


def gen_dist(apps):
    dists = {}

    keys = ['grant_value', 'sponsor', 'grant_category']
    for k in keys:
        values_agg = apps[k].value_counts().sort_index()
        values = values_agg.index
        prob = values_agg.values
        prob = prob / int(prob.sum())
        # print(prob.sum())
        dists[k]= {'probability':prob.tolist() ,'values':values.tolist()}
    return dists


def gen_lists(apps):
    category_list = apps['grant_category'].value_counts().index
    temp = apps['sponsor'].value_counts()
    sponsor_list = temp[temp>5]
    return category_list.tolist(), sponsor_list.index.tolist()


def process(dataset = 'train', createDist = True):
    base = '../data/'
    readfile = dataset+'.csv'
    writefile = dataset + '_apps_raw.csv'
    distfile = 'apps_dist_file.json'

    df = pd.read_csv(os.path.join(base, readfile), low_memory=False)
    apps = create_apps(df)
    apps.to_csv(os.path.join(base, writefile), index=False)
    print('File {} generated.'.format(writefile))

    if createDist:
        dist = gen_dist(apps)
        category_list, sponsor_list = gen_lists(apps)
        dist['lists'] = {'category':category_list,'sponsor':sponsor_list}
        with open(os.path.join(base,distfile),'w') as fp:
            json.dump(dist, fp)
        print('dist values written to {}'.format(distfile))

if __name__ == '__main__':
    process()