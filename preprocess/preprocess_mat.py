import numpy as np
import pandas as pd
from .load_mat import mat2df_zuco
import os

data_dir = './datasets/ZuCo'

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './data/zuco_preprocessed_dataframe' for local storage
tmp_path = './data/zuco_preprocessed_dataframe'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

########################################
""" Process mat: ZuCO 1.0 Task 1 - 3 """
########################################
df_zuco1 = mat2df_zuco(dataset_name='ZuCo1',
                       eeg_src_dir = data_dir + '/ZuCo1',
                       task_dir_names = ['task1-SR', 'task2-NR', 'task3-TSR'],
                       task_keys = ['task1', 'task2', 'task3'],
                       subject_keys = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', \
                                       'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'],
                       n_sentences = [400, 300, 407])

########################################
""" Process mat: ZuCO 2.0 Task 2 - 3 """
########################################
df_zuco2 = mat2df_zuco(dataset_name='ZuCo2',
                       eeg_src_dir = data_dir + '/ZuCo2',
                       task_dir_names = ['task2-NR', 'task3-TSR'],  # NOTE: we match tasks names to zuco1.0 when processing labels
                       task_keys = ['task2', 'task3'],
                       subject_keys = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', \
                                       'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', \
                                       'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'],
                       n_sentences = [349, 390])

#########################
""" Concat dataframes """
#########################
df = pd.concat([df_zuco1, df_zuco2], ignore_index = True)
print(df.shape)
print(df.columns)

#######################
""" Save to pickle """
#######################
pd.to_pickle(df, tmp_path + '/zuco_eeg_128ch_1280len.df')