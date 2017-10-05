import pandas as pd

def clearnan(x):
    m = [ s for s in x.values if s is not 'NaN']
    return m[0]



ps_df = pd.read_csv('../data/train_person_static_raw.csv')
ps_df_dd = ps_df.drop_duplicates()
ps_df_nd = ps_df_dd.groupby('Person.ID').agg(clearnan)
ps_df_nd.to_csv('../data/train_person_static_mod.csv')
ps_df2 = pd.read_csv('../data/train_person_static_mod.csv', na_filter=False)
ps_df2[['Country.of.Birth']]= ps_df2[['Country.of.Birth']].applymap(lambda x: 'Unknown' if x == '' else x)
lanmap = {"Asia Pacific": "Other",
          "Australia": "English", "Eastern Europe": "Other",
          "Great Britain": "English", "North America": "English", "Western Europe": "Other",
          "South Africa": "English", "The Americas": "Unknown", "Middle East and Africa": "Unknown",
          "Unknown": "Unknown", "New Zealand": "English"
          }


def assign_hl(x):
    if x['Home.Language'].strip() == '':
        x['Home.Language'] = lanmap[x['Country.of.Birth']]
        return x
    else:
        return x


ps_df3 = ps_df2.apply(assign_hl, axis=1)
ps_df3.set_index(['Person.ID'], inplace=True)
ps_df3.to_csv('../data/train_person_static_mod.csv')