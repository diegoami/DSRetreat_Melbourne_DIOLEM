
# Generate the RAW Files

import pandas as pd

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

def generate_by_substring(df, n):
    df['SEO.Code'] = df[['SEO.Code']].applymap(str)
    mdgs = df.groupby(['id', df['SEO.Code'].str[:n]]).sum()
    return mdgs

def generate_seo_files_raw(input_type):
    df = pd.read_csv('../data/'+input_type+'.csv')
    mdf = post_process(df)
    mdf.to_csv('../data/'+input_type+'_seo_raw.csv')
    print('Generation of raw seo data on {} done'.format(input_type))

def generate_seo_files_mod(input_type):
    mdf = pd.read_csv('../data/'+input_type+'_seo_raw.csv', index_col=None)
    mdgs = generate_by_substring(mdf, 2)
    mdgs.to_csv('../data/'+input_type+'_seo_mod.csv')
    print('Generation of mod seo data on {} done'.format(input_type))


if __name__ == '__main__':
    generate_seo_files_raw('train')
    generate_seo_files_mod('train')

    generate_seo_files_raw('test')
    generate_seo_files_mod('test')

