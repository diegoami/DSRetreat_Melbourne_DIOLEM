import numpy as np
import pandas as pd
import os.path


def split_sponsor_and_grant_category(apps):
    apps['sponsor_c'] = apps['sponsor'].str.extract('([A-Z])', expand=False)
    apps['sponsor_n'] = apps['sponsor'].str.extract('(\d+)', expand=False)
    apps['grant_category_c'] = apps['grant_category'].str.extract('([A-Z])', expand=False)
    apps['grant_category_n'] = apps['grant_category'].str.extract('(\d+)', expand=False)


def fill_nan_by_key(apps, key):
    values_agg = apps[key].value_counts().sort_index()
    values = values_agg.index
    prob = values_agg.values
    prob = prob / int(prob.sum())
    mask = apps[key].isnull()
    rands = np.random.choice(values, mask.sum(), p=prob)
    apps.loc[mask, key] = rands


def fill_nans(apps):
    keys = ['grant_value', 'sponsor', 'grant_category']
    for k in keys:
        fill_nan_by_key(apps, k)


def rebin_values(apps):
    mask = apps['grant_value'] > 400001
    apps.loc[mask, 'grant_value'] = 500000


def process(dataset = 'train'):
    base = '../data/'
    readfile = dataset + '_apps_raw.csv'
    writefile = dataset + '_apps_mod.csv'

    apps = pd.read_csv(os.path.join(base, readfile), low_memory=False)
    fill_nans(apps)
    split_sponsor_and_grant_category(apps)
    rebin_values(apps)
    apps.to_csv(os.path.join(base, writefile), index=False)
    print('File {} generated.'.format(writefile))


if __name__ == '__main__':
    process()