import pandas as pd
import os.path


def add_rfcd_code(main_table, rfcd):
    # Prepare rfcd and SEO tables for merger
    rfcd_pivoted = rfcd.pivot(
        index='id', columns='RFCD.Code', values='RFCD.Percentage').fillna(0)
    rfcd_pivoted.rename(columns=lambda x: "RFCD_"+str(x), inplace=True)
    rfcd_pivoted.reset_index(drop=False, inplace=True)

    main_table = main_table.merge(rfcd_pivoted, how='left', on='id').fillna(0)
    # Set other value to 100 and subtract any percentages appearing in other seo columns
    main_table['RFCD_OTHER'] = 100-main_table[
        [x for x in main_table.columns if x.startswith('RFCD') ]].sum(axis=1)

    # main_table.set_index('id', inplace=True)
    return main_table


def add_seo_code(main_table, seo):

    seo_pivoted  = seo.pivot(
        index='id', columns='SEO.Code', values='SEO.Percentage').fillna(0)
    seo_pivoted.rename(columns=lambda x: "SEO_"+str(x), inplace=True)
    seo_pivoted.reset_index(drop=False, inplace=True)
    main_table = main_table.merge(seo_pivoted, how='left', on='id').fillna(0)

    main_table['SEO_OTHER'] = 100-main_table[
        [x for x in main_table.columns if x.startswith('SEO') ]].sum(axis=1)
    # main_table.set_index('id', inplace=True)
    return main_table


def main(dataset='train'):
    base = '../data/'

    seof = dataset + '_seo_mod.csv'
    rfcdf = dataset + '_rfcd_mod.csv'
    appsf = dataset + '_apps_mod.csv'

    writefile = dataset+'_ml.csv'

    # load tables
    seo = pd.read_csv(os.path.join(base, seof))
    rfcd = pd.read_csv(os.path.join(base, rfcdf))
    main_table = pd.read_csv(os.path.join(base, appsf))

    #add rfcd
    main_table = add_rfcd_code(main_table, rfcd)

    #add seo
    main_table = add_seo_code(main_table, seo)

    # write to file
    main_table.to_csv(os.path.join(base,writefile))
    print('Complete table was written to {}'.format(writefile))


if __name__ == '__main__':
    main()