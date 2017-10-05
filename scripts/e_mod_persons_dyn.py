import numpy as np
import pandas as pd

# in case a person has several applications on the same day,
# we need to agg to a single line per day and person by using the following functions
day_agg_dict = {'Dept.No.' : lambda x : x.mode(),
               'Faculty.No.': lambda x : x.mode(),
               'With.PHD' : lambda x : x.max(),
                'years_in_uni' : lambda x : x.max(),
                'Number.of.Successful.Grant' : lambda x : x.max(),
                'A.':lambda x : x.max(),
                'A': lambda x : x.max(),
                'B':lambda x : x.max(),
                'C': lambda x : x.max()}

# the cummulative max will be applied on the following time series
apply_max = ['With.PHD','years_in_uni','Number.of.Successful.Grant','Number.of.Unsuccessful.Grant','A.','A','B','C']

# needed to find the file
input_type = 'train'

# load the raw data
df_p_dyn = pd.read_csv( '../data/'+ input_type + '_person_dyn_raw.csv',low_memory=False, parse_dates=['date'])

# convert the categorical variables to string
df_p_dyn.loc[:,'Person.ID'] = df_p_dyn.loc[:,'Person.ID'].astype(str)
df_p_dyn.loc[:,'Dept.No.'] = df_p_dyn.loc[:,'Dept.No.'].astype(str)
df_p_dyn.loc[:,'Faculty.No.'] = df_p_dyn.loc[:,'Faculty.No.'].astype(str)

# replace all 'nan' strings to np.nan
df_p_dyn.loc[df_p_dyn.loc[:,'Dept.No.'] == 'nan','Dept.No.'] = np.nan
df_p_dyn.loc[df_p_dyn.loc[:,'Faculty.No.'] == 'nan','Faculty.No.'] = np.nan
df_p_dyn.loc[df_p_dyn.loc[:,'With.PHD'] == 'nan','With.PHD'] = np.nan

# we set the phd coloumn to numerical Yes == 1
df_p_dyn.loc[:,'With.PHD'] = df_p_dyn.loc[:,'With.PHD'].str.strip()
df_p_dyn = df_p_dyn.replace({'With.PHD' : {'Yes' : '1'}})
df_p_dyn.loc[:,'With.PHD'] = df_p_dyn.loc[:,'With.PHD'].astype(float)

# we generate a dataframe with the unique key pairs ( date, person Id).
# this df will be filled and returned as the mod frame
df_p_dyn_mod = df_p_dyn.loc[:,['date', 'Person.ID']].drop_duplicates()

# loop over the dict to
for col, fun in day_agg_dict.items():
    # we apply the aggregation function to the coloumn
    tmp2 = df_p_dyn.groupby(['date', 'Person.ID'])[col].agg(fun)

    # in case the aggregation function returns a list and not a single element, we take the first one
    # if there were only NaN for this day and persion ID, an empty np array is returned-> we chagne it to a NaN
    tmp2 = tmp2.apply(lambda x: x[0] if (isinstance(x, np.ndarray) and len(x) > 0) else
        (np.nan if (isinstance(x, np.ndarray) and len(x) == 0) else x))

    # we merge the series with the dataframe that stores all the outcome
    df_p_dyn_mod = pd.merge(df_p_dyn_mod, tmp2.to_frame(name=col).reset_index(), how='left', on=['date', 'Person.ID'])

# we apply the cummulative maximum on all in the list apply_max
tmp2 = df_p_dyn_mod.loc[:,['Person.ID','date'] + apply_max].set_index(['Person.ID','date']).sort_index().\
    groupby(['Person.ID']).cummax().reset_index()

# we now have to replace the original columns in df_p_dyn_mod with these mnodified columns
#  keeping the columns not in apply_max
df_p_dyn_mod = pd.merge(df_p_dyn_mod.loc[:, ~df_p_dyn_mod.columns.isin(apply_max)],
                        tmp2, how='outer', on=['Person.ID','date'])

# we save the output as a csv
df_p_dyn_mod.to_csv('../data/' + input_type + '_person_dyn_mod.csv',index = False)