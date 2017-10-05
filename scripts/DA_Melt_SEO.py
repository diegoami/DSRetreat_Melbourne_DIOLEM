
# Generate the RAW Files

import pandas as pd

df = pd.read_csv('../data/unimelb_training.csv')
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

def get_seo(rfcds, code_st, per_st, id):
    seo = rfcds.melt(id_vars=[id, code_st], value_vars=[per_st]).rename(
        columns={code_st: 'SEO.Code', per_st: 'SEO.Percentage'})
    return seo

def get_melted(rfcds, id='Grant.Application.ID'):
    seo1 = get_seo(rfcds, 'SEO.Code.1', 'SEO.Percentage.1', id)
    seo2 = get_seo(rfcds, 'SEO.Code.2', 'SEO.Percentage.2', id)
    seo3 = get_seo(rfcds, 'SEO.Code.3', 'SEO.Percentage.3', id)
    seo4 = get_seo(rfcds, 'SEO.Code.4', 'SEO.Percentage.4', id)
    seo5 = get_seo(rfcds, 'SEO.Code.5', 'SEO.Percentage.5', id)
    sdcn = pd.concat([seo1, seo2, seo3, seo4, seo5]).sort_values(by=id)
    sdmelted = sdcn.loc[sdcn['value'] > 0].reset_index(drop=True)
    return sdmelted


def post_process(df):
    mdf = get_melted(df, id='id')
    mdf['SEO.Code'] = mdf[['SEO.Code']].applymap(int)
    mdf['SEO.Code'] = mdf[['SEO.Code']].applymap(str)
    mdf = mdf.drop('variable', axis=1)
    mdf = mdf.rename(columns={'value': 'SEO.Percentage'})
    mdf = mdf.set_index(['id', 'SEO.Code']).sort_index()
    return mdf


mdf_train = post_process(df_train)
mdf_train.to_csv('../data/train_seo_raw.csv')

mdf_test = post_process(df_train)
mdf_test.to_csv('../data/test_seo_raw.csv')

# # Aggregate by code


import pandas as pd

mdf_train = pd.read_csv('../data/train_seo_raw.csv', index_col=None)
mdf_test = pd.read_csv('../data/test_seo_raw.csv', index_col=None)


def generate_by_substring(df, n):
    df['SEO.Code'] = df[['SEO.Code']].applymap(str)
    mdgs = df.groupby(['id', df['SEO.Code'].str[:n]]).sum()
    return mdgs



mdgs_train = generate_by_substring(mdf_train, 2)
mdgs_train.to_csv('../data/train_seo_mod.csv')
mdgs_test = generate_by_substring(mdf_test, 2)
mdgs_test.to_csv('../data/test_seo_mod.csv')




