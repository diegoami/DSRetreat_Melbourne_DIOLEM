import pandas as pd
import os.path
from scripts.o_create_apps_raw import process as apps_process
from scripts.o_process_apps import process as apps_mod_process
from scripts.split_persons import process as split_person_process
from scripts.e_mod_persons_dyn import process as person_dyn_process
from scripts.DA_convert_person_static import convert_static_person_to_mod as convert_static_person_to_mod
from scripts.DA_Melt_Rfcd import generate_rfcd_files_mod  as rfcd_mod_process
from scripts.DA_Melt_Rfcd import generate_rfcd_files_raw  as rfcd_raw_process
from scripts.DA_Melt_SEO import generate_seo_files_mod  as seo_mod_process
from scripts.DA_Melt_SEO import generate_seo_files_raw  as seo_raw_process
from scripts.e_mod_roles import process  as role_mod_process
from scripts.e_generate_success_features import process as suc_process



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

    #role mod
    role_mod_process(trainf)
    role_mod_process(testf)

    # person dynamic_mod
    person_dyn_process(trainf)
    person_dyn_process(testf, merge_with=trainf)

    # person_static_raw => person_static_mod
    convert_static_person_to_mod(trainf)
    convert_static_person_to_mod(testf)

    # generation of RFCD_raw
    rfcd_raw_process(trainf)
    rfcd_raw_process(testf)

    # add generation of RFCD_mod
    rfcd_mod_process(trainf)
    rfcd_mod_process(testf)

    #add generation of SEO_raw
    seo_raw_process(trainf)
    seo_raw_process(testf)

    #add generation of SEO_mod
    seo_mod_process(trainf)
    seo_mod_process(testf)

    #add success rate
    suc_process(trainf)
    suc_process(testf, merge_with = trainf)

    print('Finished generating all raw and mod tables !!!')


if __name__ == '__main__':
    main()