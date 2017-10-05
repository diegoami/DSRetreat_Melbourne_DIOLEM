import pandas as pd

def clearnan(x):
    m = [ s for s in x.values if s is not 'NaN']
    return m[0]

lanmap = {"Asia Pacific": "Other",
          "Australia": "English", "Eastern Europe": "Other",
          "Great Britain": "English", "North America": "English", "Western Europe": "Other",
          "South Africa": "English", "The Americas": "Unknown", "Middle East and Africa": "Unknown",
          "Unknown": "Unknown", "New Zealand": "English"
          }




def convert_df(ps_df, tmp_file):
    ps_df_dd = ps_df.drop_duplicates()
    ps_df_nd = ps_df_dd.groupby('Person.ID').agg(clearnan)
    ps_df_nd.to_csv(tmp_file)
    ps_df2 = pd.read_csv(tmp_file, na_filter=False)
    ps_df2[['Country.of.Birth']] = ps_df2[['Country.of.Birth']].applymap(lambda x: 'Unknown' if x == '' else x)

    def assign_hl(x):
        if x['Home.Language'].strip() == '':
            x['Home.Language'] = lanmap[x['Country.of.Birth']]
            return x
        else:
            return x

    ps_df3 = ps_df2.apply(assign_hl, axis=1)
    ps_df3.set_index(['Person.ID'], inplace=True)
    return ps_df3

ps_df_train = pd.read_csv('../data/train_person_static_raw.csv')
ps_df_train_conv = convert_df(ps_df_train, '../data/train_person_static_temp.csv' )
ps_df_train_conv.to_csv('../data/train_person_static_mod.csv')

#ps_df_test = pd.read_csv('../data/test_person_static_raw.csv')
#ps_df_test_conv = convert_df(ps_df_test, '../data/test_person_static_temp.csv')
#ps_df_test_conv.to_csv('../data/test_person_static_mod.csv')
