import pandas as pd
import os.path


def main(dataset='train'):
    base = '../data/'

    seof = dataset + '_seo_mod.csv'
    rfcdf = dataset + '_rfcd_mod.csv'
    appsf = dataset + '_apps_mod.csv'

    writefile = dataset+'_ml.csv'

    # load tables
    seo = pd.read_csv(os.path.join(base, seof))
    rfcd = pd.read_csv(os.path.join(base, rfcdf))
    app_train = pd.read_csv(os.path.join(base, appsf))

    # Prepare rfcd and SEO tables for merger
    rfcd_pivoted = rfcd.pivot(
        index='id', columns='RFCD.Code', values='RFCD.Percentage').fillna(0)
    seo_pivoted  = seo.pivot(
        index='id', columns='SEO.Code', values='SEO.Percentage').fillna(0)

    rfcd_pivoted.rename(columns=lambda x: "RFCD_"+str(x), inplace=True)
    seo_pivoted.rename(columns=lambda x: "SEO_"+str(x), inplace=True)



    app_rfcd = app_train.join(rfcd_pivoted, how='outer').fillna(0)
    app_rfcd_seo = app_rfcd.join(seo_pivoted, how='outer').fillna(0)

    # Set other value to 100 and subtract any percentages appearing in other seo columns
    app_rfcd_seo['RFCD_OTHER'] = 100-app_rfcd_seo[
        [x for x in app_rfcd_seo.columns if x.startswith('RFCD') ]].sum(axis=1)

    app_rfcd_seo['SEO_OTHER'] = 100-app_rfcd_seo[
        [x for x in app_rfcd_seo.columns if x.startswith('SEO') ]].sum(axis=1)

    app_rfcd_seo.set_index('id', inplace=True)


    # write to file
    app_rfcd_seo.to_csv(os.path.join(base,writefile))
    print('Complete table was written to {}'.format(writefile))


if __name__ == '__main__':
    main()