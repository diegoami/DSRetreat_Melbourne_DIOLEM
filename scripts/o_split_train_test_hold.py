import numpy as np
import pandas as pd
import os.path

def main():
    #column renaming dictionary
    ren = {'Grant.Application.ID': 'id',
           'Grant.Status': 'granted',
           'Sponsor.Code': 'sponsor',
           'Grant.Category.Code': 'grant_category',
           'Contract.Value.Band...see.note.A': 'grant_value',
           'Start.date': 'date',
           'No..of.Years.in.Uni.at.Time.of.Grant': 'years_in_uni'
           }

    base = '../data/'
    fdata = 'unimelb_training.csv'
    ftest = 'training2_ids.csv'
    fhold = 'testing_ids.csv'
    savetrain = 'train.csv'
    savetest = 'test.csv'
    savehold = 'hold.csv'



    df = pd.read_csv(os.path.join(base, fdata), low_memory=False)
    id_test = pd.read_csv(os.path.join(base, ftest), low_memory=False)
    id_hold = pd.read_csv(os.path.join(base, fhold), low_memory=False)

    #Weird additional column ?
    df.drop(['Unnamed: 251'], axis=1, inplace=True)

    #date to datetime
    df.rename(columns=ren, inplace=True)

    #generate train and hold
    test = df.loc[df['id'].isin(id_test.values.flatten()), :]
    hold = df.loc[df['id'].isin(id_hold.values.flatten()), :]

    #generate training (hold test removed)
    train = df.copy()
    drop_ids = np.r_[id_test.values.flatten(), id_hold.values.flatten()]
    mask = train['id'].isin(drop_ids)
    train.drop(
        (train.loc[mask, 'id']).index, axis=0, inplace=True)

    print('size of test data: {}\nsize of hold data: {}\nsize of train data: {}'.format(
        test.shape, hold.shape, train.shape))

    #Save everything
    base = '../data/'

    train.to_csv(os.path.join(base, savetrain), index=False)
    test.to_csv(os.path.join(base, savetest), index=False)
    hold.to_csv(os.path.join(base, savehold), index=False)


if __name__ == '__main__':
    main()