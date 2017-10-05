import numpy as np
import pandas as pd

input_type = 'train'
df = pd.read_csv('../data/' + input_type + '.csv', low_memory=False)

persons = pd.concat( [df.iloc[:, 0],  df.loc[:, 'Person.ID.1':'C.15']], axis = 1)

#we flatten the persons subtable
#all columns with persons informatin is repeated 15 times
n_persons = 15

#this gives us the following number of coloumns per person (also 15):
n_persons_cols = (len(persons.columns) - 1) // n_persons

#we generate a list the with ID and the field names
col_names = [persons.columns[0]]
[col_names.append(i[:-2]) for i in persons.columns[1:n_persons_cols+1]]

# and use this list to make an empty dataframe
df_p = pd.DataFrame(columns = col_names)

#we loop over all persons 1..15 and take the respective fields (ie columns) and append them
for i in range(n_persons):
    #coloumn index for first coloumn of person i
    k = 1 + (i*n_persons_cols)
    #coloumn index for last coloumn of person i
    l = 1 + ((i+1)*n_persons_cols)
    #get a temp table with the desired info
    tmp = pd.concat( [persons.iloc[:,0],  persons.iloc[:,k:l]], axis = 1)
    #rename the table to make the UNION ALL opertion work
    tmp.columns = df_p.columns
    df_p = pd.concat([df_p, tmp],axis = 0)

# remove lines with person ID AND Role empty
df_p = df_p.loc[~(df_p.loc[:,'Person.ID'].isnull() & df_p.loc[:,'Role'].isnull()),:]

# we create a new table called externals with the  columns grant_application_id and has_[external role]
# external roles are stored as person without ID, therefore the following command identifies them:
# df_p.loc[df_p.loc[:,'Person.ID'].isnull(),'Role'].unique()
# of course they can also be identified by name (ie external advisor..)
# it might be possible that a different role has a missing person ID filed in the testing date,
# therefore we hardcoded it: EXT_CHIEF_INVESTIGATOR', 'STUD_CHIEF_INVESTIGATOR', 'STUDRES', 'EXTERNAL_ADVISOR'
ext_list = ['EXT_CHIEF_INVESTIGATOR', 'STUD_CHIEF_INVESTIGATOR', 'STUDRES', 'EXTERNAL_ADVISOR']
mask = df_p.Role.isin(ext_list)
df_ext = df_p.loc[mask,:]
#we only want to keep the grant application id and the role
df_ext = df_ext.loc[:,['id','Role']]

df_p = df_p.loc[~mask,:]

# next step is to pivot the table so that we have one line per grant applications ID
df_ext['dummy'] = 1
df_ext = df_ext.pivot_table(index ='id', columns='Role', values = 'dummy', aggfunc=np.sum )

# we restore the column ID by restting the index
df_ext = df_ext.reset_index()

# we write the externals table to a csv file
df_ext.to_csv( '../data/'+ input_type + '_externals_raw.csv',index = False)

# this should be 0
if df_p.loc[:,'Person.ID'].isnull().sum() != 0:
    raise(ValueError, 'lines with missing Person IDs after removing externals')

# we can now continue with the person table
# we make all categorial variables to strings
# numerics are first converted to integers, otherwise we will have .0 at the end
df_p[['Person.ID']] = df_p[['Person.ID']].astype(int).astype(str)
df_p[['Dept.No.','Faculty.No.']] = df_p[['Dept.No.','Faculty.No.']].astype(str)

# next columns to work on is the years spend in the uni
# we rename the column
df_p = df_p.rename(columns={'No..of.Years.in.Uni.at.Time.of.Grant': 'years_in_uni'})
# we use a dict to remap the values
dict_years_in_uni = {'Less than 0': 0.,
    '>=0 to 5': 1.,
    '>5 to 10': 2.,
    '>10 to 15': 3.,
    'more than 15': 4.}
df_p = df_p.replace({'years_in_uni' : dict_years_in_uni})

# we split it in two dataframes: static and dynamic
static_columns = ['Person.ID','Year.of.Birth','Country.of.Birth','Home.Language']
df_p_static = df_p.loc[:,static_columns]
df_p_static.to_csv(  '../data/'+ input_type + '_person_static_raw.csv',index = False)

# in the dynamic df comes everything that is not in the static + the key
df_p_dyn = df_p.loc[:,( (~(df_p.columns.isin(static_columns))) | df_p.columns.isin(['id','Person.ID']))]
# we only remove the Role, since we will have a separate table for this
df_p_dyn = df_p_dyn.loc[:,~df_p_dyn.columns.isin(['Role']) ]

# we join the date of teh application id
df_p_dyn  = pd.merge( df.loc[:,['id','date']], df_p_dyn, on = 'id')

# we remove the id column, rows shoud be identified by date and person id.
df_p_dyn = df_p_dyn.loc[:,~df_p_dyn.columns.isin(['id']) ]

# and write the output
df_p_dyn.to_csv(  '../data/'+ input_type + '_person_dyn_raw.csv',index = False)

#we extract the roles and save them to a csv file
roles = df_p.loc[:, ['id', 'Person.ID', 'Roles']]

roles.to_csv(  '../data/'+ input_type + '_roles_raw.csv',index = False)