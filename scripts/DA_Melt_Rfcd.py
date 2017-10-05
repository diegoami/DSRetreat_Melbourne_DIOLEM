# Generate the RAW Files

import pandas as pd
df = pd.read_csv('../data/unimelb_training.csv')

def get_melted(rfcds, id='Grant.Application.ID'):
    rcd1 = rfcds.melt(id_vars=[id, 'RFCD.Code.1'], value_vars=['RFCD.Percentage.1']).rename(
        columns={'RFCD.Code.1': 'RFCD.Code', 'RFCD.Percentage.1': 'RFCD.Percentage'})
    rcd2 = rfcds.melt(id_vars=[id, 'RFCD.Code.2'], value_vars=['RFCD.Percentage.2']).rename(
        columns={'RFCD.Code.2': 'RFCD.Code', 'RFCD.Percentage.2': 'RFCD.Percentage'})
    rcd3 = rfcds.melt(id_vars=[id, 'RFCD.Code.3'], value_vars=['RFCD.Percentage.3']).rename(
        columns={'RFCD.Code.3': 'RFCD.Code', 'RFCD.Percentage.3': 'RFCD.Percentage'})
    rcd4 = rfcds.melt(id_vars=[id, 'RFCD.Code.4'], value_vars=['RFCD.Percentage.4']).rename(
        columns={'RFCD.Code.4': 'RFCD.Code', 'RFCD.Percentage.4': 'RFCD.Percentage'})
    rdcn = pd.concat([rcd1, rcd2, rcd3, rcd4]).sort_values(by=id)
    rdmelted = rdcn.loc[rdcn['value'] > 0].reset_index(drop=True)
    return rdmelted


rfcds = pd.concat([df.iloc[:, 0], df.iloc[:, 6:16]], axis=1)
rdmelted = get_melted(rfcds)
rdmelted.to_csv('../data/rtcd_raw.csv')

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')


def post_process(df):
    mdf = get_melted(df, id='id')
    mdf['RFCD.Code'] = mdf[['RFCD.Code']].applymap(int)
    mdf['RFCD.Code'] = mdf[['RFCD.Code']].applymap(str)
    mdf = mdf.drop('variable', axis=1)
    mdf = mdf.rename(columns={'value': 'RFCD.Percentage'})
    mdf = mdf.set_index(['id', 'RFCD.Code']).sort_index()
    return mdf

mdf_train = post_process(df_train)
mdf_train.to_csv('../data/train_rfcd_raw.csv')

mdf_test = post_process(df_train)
mdf_test.to_csv('../data/test_rfcd_raw.csv')


mdf_train = pd.read_csv('../data/train_rfcd_raw.csv', index_col=None)
mdf_test = pd.read_csv('../data/test_rfcd_raw.csv', index_col=None)


def generate_by_substring(df, n):
    df['RFCD.Code'] = df[['RFCD.Code']].applymap(str)
    mdgs = df.groupby(['id', df['RFCD.Code'].str[:n]]).sum()
    return mdgs


mdgs_train = generate_by_substring(mdf_train, 2)
mdgs_train.to_csv('../data/train_rfcd_mod.csv')
mdgs_test = generate_by_substring(mdf_test, 2)
mdgs_test.to_csv('../data/test_rfcd_mod.csv')


# In[ ]:



