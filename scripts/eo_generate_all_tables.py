import pandas as pd
import os.path
from scripts.o_create_apps_raw import process as apps_process
from scripts.o_process_apps import process as apps_mod_process
from scripts.split_persons import process as split_person_process


def main():
    base = '../data/'
    trainf = 'train.csv'
    testf  = 'test.csv'

    apps_process('train')
    apps_process('test', createDist=False)

    apps_mod_process('train')
    apps_mod_process('test')

    split_person_process('train')



if __name__ == '__main__':
    main()