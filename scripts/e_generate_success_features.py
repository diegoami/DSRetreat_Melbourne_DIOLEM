import pandas as pd


def gen_s(df_filter, df_roles,df_main):
    df_dates = df_main.loc[:, ['id','date']].drop_duplicates()
    df_filter.columns = ['id', 'f_value']

    result = pd.DataFrame(columns=['s_count', 's_rate'])

    for id in df_filter.id.unique():
        # print(id)
        team = list(df_roles.loc[df_roles.id == id, 'Person.ID'])

        # this is a list of all applications, also future ones
        all_team_apps = pd.Series(df_roles.loc[df_roles.loc[:, 'Person.ID'].isin(team), 'id'].unique())

        # we now need to filter for previous applications of the team
        date_this_app = df_dates.loc[df_dates.id == id, 'date']
        ids_all_prev_apps = list(df_dates.loc[df_dates.date < date_this_app.iloc[0], 'id'])
        prev_team_apps = list(all_team_apps[all_team_apps.isin(ids_all_prev_apps)])

        # of those previous applications, we now keep only the ones that were in the same category
        # ie same sponsor, same rfcd
        current_filter = list(df_filter.loc[df_filter.id == id, 'f_value'].unique())
        prev_team_apps_filter = list(df_filter. \
                                     loc[df_filter.f_value.isin(current_filter) & df_filter.id.isin(prev_team_apps), 'id'])

        granted_filter = df_main.loc[df_main.id.isin(prev_team_apps_filter), 'granted']

        tmp = pd.DataFrame({'s_count': granted_filter.sum(),
                            's_rate': granted_filter.mean()}, index=[id])
        result = result.append(tmp)

    result.index.names = ['id']
    result = result.reset_index()
    return result

def gen_filter_classifications(df_main, rfcd_level, class_type):
    class_level_dict = {0: 10000,
                       1: 100,
                       2: 1}
    apps_class = df_main.set_index('id').filter(regex='^' + class_type + '.Code.'). \
                    reset_index().melt('id').query("value != 0."). \
                    assign(classification=lambda df: df.value // class_level_dict[rfcd_level]).sort_values('id'). \
                    reset_index(drop=True).loc[:, ['id', 'classification']].drop_duplicates()

    return apps_class

def gen_filter_numeric(df_main, col_name):
    df_filter = df_main.set_index('id').filter(regex='^' + col_name). \
                    reset_index().melt('id').query("value != 0."). \
                    sort_values('id'). \
                    reset_index(drop=True).loc[:, ['id', 'value']].drop_duplicates()
    return df_filter


def process(input_type, merge_with = None):
    # we also need the combination (id,date)
    df_main = pd.read_csv('../data/' + input_type + '.csv', low_memory=False, parse_dates=['date'])
    df_roles = pd.read_csv('../data/' + input_type + '_role_mod.csv', low_memory=False)

    if merge_with:
        keep_key = df_main.loc[:,['id']].drop_duplicates()
        df_main.append(pd.read_csv( '../data/'+ merge_with + '.csv',low_memory=False, parse_dates=['date']))
        df_roles.append(pd.read_csv('../data/' + merge_with + '_role_mod.csv', low_memory=False))

    #we create an df that has only the id coloumn
    df_out = df_main.loc[:,['id']]

    # we loop over all rfcd levels and save the success rate and sum of the teams
    for j in range(3):
        print('rfcd ' + str(j))
        df_filter = gen_filter_classifications(df_main, j,'RFCD')
        tmp_res = gen_s(df_filter, df_roles, df_main)
        tmp_res.columns =  ['rfcd_l' + str(j) + '_' + x  if x !='id' else x  for x in tmp_res.columns]
        df_out = df_out.merge(tmp_res, how='left', on='id')

    # we loop over all seo levels and save the success rate and sum of the teams
    for j in range(3):
        print('seo ' + str(j))
        df_filter = gen_filter_classifications(df_main, j, 'SEO')
        tmp_res = gen_s(df_filter, df_roles, df_main)
        tmp_res.columns = ['seo_l' + str(j) + '_' + x if x != 'id' else x for x in tmp_res.columns]
        df_out = df_out.merge(tmp_res, how='left', on='id')

    # department filter
    df_filter = gen_filter_numeric(df_main, 'Dept.No.')
    tmp_res = gen_s(df_filter, df_roles, df_main)
    tmp_res.columns = ['Dept.No.' + '_' + x if x != 'id' else x for x in tmp_res.columns]
    df_out = df_out.merge(tmp_res, how='left', on='id')

    #faculty
    df_filter = gen_filter_numeric(df_main, 'Faculty.No.')
    tmp_res = gen_s(df_filter, df_roles, df_main)
    tmp_res.columns = ['Faculty.No.' + '_' + x if x != 'id' else x for x in tmp_res.columns]
    df_out = df_out.merge(tmp_res, how='left', on='id')

    # sponsor
    df_filter = df_main.loc[:,['id','sponsor']]
    tmp_res = gen_s(df_filter, df_roles, df_main)
    tmp_res.columns = ['sponsor' + '_' + x if x != 'id' else x for x in tmp_res.columns]
    df_out = df_out.merge(tmp_res, how='left', on='id')

    # grant_category
    df_filter = df_main.loc[:, ['id', 'grant_category']]
    tmp_res = gen_s(df_filter, df_roles, df_main)
    tmp_res.columns = ['grant_category' + '_' + x if x != 'id' else x for x in tmp_res.columns]
    df_out = df_out.merge(tmp_res, how='left', on='id')

    if merge_with:
        df_out.merge(keep_key, how='inner', on=['id'])

    filename = '../data/' + input_type + '_success_features.csv'
    df_out.to_csv(filename, index=False)
    print('I ran and generated {}'.format(filename))

if __name__ == '__main__':
    process(input_type='train')






