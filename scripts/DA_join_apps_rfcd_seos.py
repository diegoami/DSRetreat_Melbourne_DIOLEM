import pandas as pd
seo_train = pd.read_csv('../data/train_seo_mod.csv')
rfcd_train = pd.read_csv('../data/train_rfcd_mod.csv')
app_train = pd.read_csv('../data/train_apps_mod.csv')

app_rfcd_train = app_train.merge(rfcd_train, on='id', how='outer')

rfcd_train_pivoted = rfcd_train.pivot( index='id', columns='RFCD.Code', values='RFCD.Percentage').fillna(0)
seo_train_pivoted  = seo_train.pivot( index='id', columns='SEO.Code', values='SEO.Percentage').fillna(0)
rfcd_train_pivoted.rename(columns=lambda x: "RFCD_"+str(x), inplace=True)
seo_train_pivoted.rename(columns=lambda x: "SEO_"+str(x), inplace=True)
app_rfcd_train = app_train.join(rfcd_train_pivoted, how='outer').fillna(0)

app_rfcd_seo_train = app_rfcd_train.join(seo_train_pivoted, how='outer').fillna(0)

app_rfcd_seo_train['RFCD_OTHER'] = 100-app_rfcd_seo_train[[x for x in app_rfcd_seo_train.columns if x.startswith('RFCD') ]].sum(axis=1)
app_rfcd_seo_train['SEO_OTHER'] = 100-app_rfcd_seo_train[[x for x in app_rfcd_seo_train.columns if x.startswith('SEO') ]].sum(axis=1)
app_rfcd_seo_train.set_index('id', inplace=True)
app_rfcd_seo_train.to_csv('../data/train_apps_rfcd_seo_mod.csv')
