import pandas as pd
import os.path
from scripts.o_create_apps_raw import process as apps_process
from scripts.o_process_apps import process as apps_mod_process
from scripts.split_persons import process as split_person_process
from scripts.e_mod_persons_dyn import process as person_dyn_process


def main():
    base = '../data/'
    trainf = 'train'
    testf  = 'test'

    #generates _apps_raw
    apps_process(trainf)
    apps_process(testf, createDist=False)

    # generates _apps_mod
    apps_mod_process(trainf)
    apps_mod_process(testf)

    # generated _externals_raw, roles_raw persons static_raw person dynamic_raw
    split_person_process(trainf)
    split_person_process(testf)

    # person dynamic_mod
    person_dyn_process(trainf)
    person_dyn_process(testf, merge_with=trainf)

    #Diego ToDo:
    #add person_static_raw => person_static_mod
    #add generation of SEO_raw
    #add generation of SEO_mod
    #add generation of RFCD_raw
    #add generation of RFCD_mod

    print('Finished generating all raw and mod tables !!!')

if __name__ == '__main__':
    main()