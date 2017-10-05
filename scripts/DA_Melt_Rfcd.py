# Generate the RAW Files

import pandas as pd


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

def generate_by_substring(df, n):
    df['RFCD.Code'] = df[['RFCD.Code']].applymap(str)
    mdgs = df.groupby(['id', df['RFCD.Code'].str[:n]]).sum()
    return mdgs


def post_process(df):
    mdf = get_melted(df, id='id')
    mdf['RFCD.Code'] = mdf[['RFCD.Code']].applymap(int)
    mdf['RFCD.Code'] = mdf[['RFCD.Code']].applymap(str)
    mdf = mdf.drop('variable', axis=1)
    mdf = mdf.rename(columns={'value': 'RFCD.Percentage'})
    mdf = mdf.set_index(['id', 'RFCD.Code']).sort_index()
    return mdf

def generate_rfcd_files_raw(input_type):
    df = pd.read_csv('../data/'+input_type+'.csv')
    mdf = post_process(df)
    mdf.to_csv('../data/'+input_type+'_rfcd_raw.csv')
    print('Generation of raw rcd data on {} done'.format(input_type))

def generate_rfcd_files_mod(input_type):
    mdf = pd.read_csv('../data/'+input_type+'_rfcd_raw.csv', index_col=None)
    mdgs = generate_by_substring(mdf, 2)
    mdgs.to_csv('../data/'+input_type+'_rfcd_mod.csv')
    print('Generation of mod rcd data on {} done'.format(input_type))

if __name__ == '__main__':
    generate_rfcd_files_raw('train')
    generate_rfcd_files_mod('train')

    generate_rfcd_files_raw('test')
    generate_rfcd_files_mod('test')

"""df_train = pd.read_csv('../data/train.csv')
mdf_train = post_process(df_train)
mdf_train.to_csv('../data/train_rfcd_raw.csv')
mdf_train = pd.read_csv('../data/train_rfcd_raw.csv', index_col=None)
mdgs_train = generate_by_substring(mdf_train, 2)
mdgs_train.to_csv('../data/train_rfcd_mod.csv')

df_test = pd.read_csv('../data/test.csv')
mdf_test = post_process(df_train)
mdf_test.to_csv('../data/test_rfcd_raw.csv')
mdf_test = pd.read_csv('../data/test_rfcd_raw.csv', index_col=None)
mdgs_test = generate_by_substring(mdf_test, 2)
mdgs_test.to_csv('../data/test_rfcd_mod.csv')"""


# In[ ]:



