import pandas as pd
import os.path
import numpy as np

#TODO
#smarter way to deal with external

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


def aggregate_people(df_roles, df_p_dyn, df_p_static, df_dates):

    feature_dict = {'Year.of.Birth': {'mean_year_of_birth': lambda x: x.mean(),
                                         'max_year_of_birth': lambda x: x.max()},
                        'Role': {'n_internals': lambda x: x.size},
                       'Dept.No.': {'Dept.No.': lambda x: x.mode()},
                       'Faculty.No.': {'Faculty.No.': lambda x: x.mode()},
                       'With.PHD': {'With.PHD': lambda x: x.sum()},
                       'years_in_uni': {'mean_years_in_uni': lambda x: x.mean(),
                                        'max_years_in_uni': lambda x: x.max()},
                       'Number.of.Successful.Grant': {'Number.of.Successful.Grant': lambda x: x.sum()},
                       'Number.of.Unsuccessful.Grant': {'Number.of.Unsuccessful.Grant': lambda x: x.sum()},
                       'A.': {'A.': lambda x: x.sum()},
                       'A': {'A': lambda x: x.sum()},
                       'B': {'B': lambda x: x.sum()},
                       'C': {'C': lambda x: x.sum()}}
    # 'Country.of.Birth': {'Country.of.Birth': lambda x: x.mode()},
    # 'Home.Language': {'Home.Language': lambda x: x.mode()},

    # we also need the combination (id,date)
    # df_dates = pd.read_csv('../data/' + input_type + '.csv', low_memory=False, parse_dates=['date']).loc[:,
    #            ['id', 'date']].drop_duplicates()

    persons = df_dates.merge(df_roles, how='left', on=['id']). \
        merge(df_p_static, how='left', on=['Person.ID']). \
        merge(df_p_dyn, how='left', on=['date', 'Person.ID'])

    # we generate a dataframe with the uniqe key pairs ( date, person Id).
    # this df will be filled and returned as the mod frame
    df_out = df_dates.loc[:, ['id']].drop_duplicates()

    for col_in, fun_dict in feature_dict.items():
        for col_out, fun in fun_dict.items():
            # we apply the aggregation function to the coloumn
            tmp2 = persons.groupby(['id'])[col_in].agg(fun)

            # in case the aggregation function returns a list and not a single element, we take the first one
            # if there were only NaN for this day and persion ID, an empty np array is returned-> we chagne it to a NaN
            tmp2 = tmp2.apply(lambda x: x[0] if (isinstance(x, np.ndarray) and len(x) > 0) else \
                (np.nan if (isinstance(x, np.ndarray) and len(x) == 0) else x))

            # we merge the series with the dataframe that stores all the outcome
            df_out = pd.merge(df_out, tmp2.to_frame(name=col_out).reset_index(), how='left', on=['id'])
    return df_out

def add_agg_people(main_table, aggp):
    main_table = main_table.merge(aggp, on='id', how='left')
    main_table.fillna(0,inplace = True)
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

def add_externals(main_table, ext):
    main_table = main_table.merge(ext, on='id', how='left')
    main_table.fillna(0, inplace=True)

    return main_table


def treat_date(main_table):
    # get the month
    main_table['month'] = main_table['date'].dt.month

    # get the timediff in days from 2005 - 01 - 01
    main_table['date'] = (main_table['date'] - pd.Timestamp('2005-01-01')).dt.days
    return main_table


def main(dataset='train'):
    base = '../data/'

    seof = dataset + '_seo_mod.csv'
    rfcdf = dataset + '_rfcd_mod.csv'
    appsf = dataset + '_apps_mod.csv'
    extf   = dataset + '_externals_raw.csv'

    rolef = dataset +'_role_mod.csv'
    persondf = dataset + '_person_dyn_mod.csv'
    personsf = dataset +'_person_static_mod.csv'

    writefile = dataset+'_ml.csv'

    # load tables
    seo = pd.read_csv(os.path.join(base, seof))
    rfcd = pd.read_csv(os.path.join(base, rfcdf))
    ext = pd.read_csv(os.path.join(base, extf))

    role = pd.read_csv(os.path.join(base, rolef))
    persond = pd.read_csv(os.path.join(base, persondf), parse_dates=['date'])
    persons = pd.read_csv(os.path.join(base, personsf))

    main_table = pd.read_csv(os.path.join(base, appsf), parse_dates=['date'])
    dates = main_table[['id','date']]
    person_agg = aggregate_people(role, persond, persons, dates)


    #add rfcd
    main_table = add_rfcd_code(main_table, rfcd)
    #add seo
    main_table = add_seo_code(main_table, seo)

    #add ext
    main_table = add_externals(main_table, ext)

    #add agg
    main_table = add_agg_people(main_table, person_agg)

    main_table = treat_date(main_table)
    # write to file
    main_table.to_csv(os.path.join(base,writefile), index=False)
    print(main_table.head())
    print('Complete table was written to {}'.format(writefile))


if __name__ == '__main__':
    main()
    main(dataset='test')
