import numpy as np
import pandas as pd

agg_roles_dict={'Role':  lambda x : x.mode() }
df_rol_raw = pd.read_csv('../data/' + input_type + '_roles_raw.csv', low_memory=False)

# we generate a dataframe with the uniqe key pairs ( date, person Id).
# this df will be filled and returned as the mod frame
df_rol_mod = df_rol_raw.loc[:, ['id', 'Person.ID']].drop_duplicates()

for col, fun in agg_roles_dict.items():
    # we apply the aggregation function to the coloumn
    tmp2 = df_rol_raw.groupby(['id', 'Person.ID'])[col].agg(fun)

    # in case the aggregation function returns a list and not a single element, we take the first one
    # if there were only NaN for this day and persion ID, an empty np array is returned-> we chagne it to a NaN
    tmp2 = tmp2.apply(lambda x: x[0] if (isinstance(x, np.ndarray) and len(x) > 0) else \
        (np.nan if (isinstance(x, np.ndarray) and len(x) == 0) else x))

    # we merge the series with the dataframe that stores all the outcome
    df_rol_mod = pd.merge(df_rol_mod, tmp2.to_frame(name=col).reset_index(), how='left', on=['id', 'Person.ID'])

df_rol_mod.to_csv('../data/' + input_type + '_role_mod.csv',index = False)