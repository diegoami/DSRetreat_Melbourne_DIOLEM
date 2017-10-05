import pandas as pd
seo_train = pd.read_csv('../data/train_seo_mod.csv')
rfcd_train = pd.read_csv('../data/train_rfcd_mod.csv')
app_train = pd.read_csv('../data/train_apps_mod.csv')

seo_test = pd.read_csv('../data/test_seo_mod.csv')
rfcd_test = pd.read_csv('../data/test_rfcd_mod.csv')
app_test = pd.read_csv('../data/test_apps_mod.csv')


def generate_table(seo, rfcd, app):
    rfcd_pivoted = rfcd.pivot(index='id', columns='RFCD.Code', values='RFCD.Percentage').fillna(0)
    seo_pivoted = seo.pivot(index='id', columns='SEO.Code', values='SEO.Percentage').fillna(0)
    rfcd_pivoted.rename(columns=lambda x: "RFCD_" + str(x), inplace=True)
    seo_pivoted.rename(columns=lambda x: "SEO_" + str(x), inplace=True)
    app_rfcd = app.join(rfcd_pivoted, how='left').fillna(0)
    app_rfcd_seo = app_rfcd.join(seo_pivoted, how='left').fillna(0)
    app_rfcd_seo['RFCD_OTHER'] = 100 - app_rfcd_seo[
        [x for x in app_rfcd_seo.columns if x.startswith('RFCD')]].sum(axis=1)
    app_rfcd_seo['SEO_OTHER'] = 100 - app_rfcd_seo[
        [x for x in app_rfcd_seo.columns if x.startswith('SEO')]].sum(axis=1)
    app_rfcd_seo.set_index('id', inplace=True)
    return app_rfcd_seo


app_rfcd_seo_train = generate_table(seo_train, rfcd_train, app_train)
app_rfcd_seo_train.to_csv('../data/train_apps_rfcd_seo_mod.csv')

app_rfcd_seo_test = generate_table(seo_test, rfcd_test, app_test)
app_rfcd_seo_test.to_csv('../data/test_apps_rfcd_seo_mod.csv')
