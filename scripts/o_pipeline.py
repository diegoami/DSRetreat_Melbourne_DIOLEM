import numpy as np
import pandas as pd
import os.path

from scripts.eo_generate_all_tables import main as gen_tab_mod


def main():
    eq = '='*50
    print(eq + '\ngenerating all the tables'+eq)
    gen_tab_mod()


if __name__ == '__main__':
    main()


