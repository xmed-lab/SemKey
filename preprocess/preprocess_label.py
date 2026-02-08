import numpy as np
import pandas as pd
import os

dataset_dir = './datasets/ZuCo'
data_dir_zuco1 = dataset_dir + '/ZuCo1'
zuco1_task1_lbl_path = data_dir_zuco1 + '/revised_csv/sentiment_labels_task1.csv'
zuco1_task2_lbl_path = data_dir_zuco1 + '/revised_csv/relations_labels_task2.csv'
zuco1_task3_lbl_path = data_dir_zuco1 + '/revised_csv/relations_labels_task3.csv'
data_dir_zuco2 = dataset_dir + '/ZuCo2/task_materials'

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './data/zuco_preprocessed_dataframe' for local storage
tmp_path = './data/zuco_preprocessed_dataframe'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

########################
""" ZuCO 1.0 task 1 """
########################
df11_raw = pd.read_csv(zuco1_task1_lbl_path, 
                       sep=';', header=0,  skiprows=[1], encoding='utf-8',
                       dtype={'sentence': str, 'control': str, 'sentiment_label':str})
# print(df11_raw)
# n_row, n_column = df11_raw.shape
df11 = df11_raw.rename(columns={'sentence': 'raw text', 
                            'sentiment_label': 'raw label'})
df11 = df11.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
                      
df11['dataset'] =  ['ZuCo1'] * df11.shape[0]  # each item is init as a tuple with len==1 for easy extension
df11['task'] =  ['task1'] * df11.shape[0]
# drop unused column
df11 = df11.drop(['control'], axis = 1)

# print(df11.shape, df11.columns)
# print(df11['raw text'].nunique())

########################
""" ZuCO 1.0 task 2 """
########################
df12_raw = pd.read_csv(zuco1_task2_lbl_path, 
                       sep=',', header=0, encoding='utf-8',
                       dtype={'sentence': str,'control': str,'relation_types':str})
# n_row, n_column = df12_raw.shape
df12 = df12_raw.rename(columns={'sentence': 'raw text', 
                                'relation_types': 'raw label'})
df12 = df12.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
df12['dataset'] =  ['ZuCo1'] * df12.shape[0]
df12['task'] =  ['task2'] * df12.shape[0]
# drop unused column
df12 = df12.drop(['control'], axis = 1)

# print(df12.shape, df12.columns)
# print(df12['raw text'].nunique())

########################
""" ZuCO 1.0 task 3 """
########################
df13_raw = pd.read_csv(zuco1_task3_lbl_path, 
                       sep=';', header=0, encoding='utf-8', 
                       dtype={'sentence': str, 'relation-type':str})
df13 = df13_raw.rename(columns={'sentence': 'raw text', 
                            'relation-type': 'raw label'})
df13 = df13.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
df13['dataset'] =  ['ZuCo1'] * df13.shape[0]
df13['task'] =  ['task3'] * df13.shape[0]
# drop unused column
df13 = df13.drop(['control'], axis = 1)

# print(df13.shape, df13.columns)
# print(df13['raw text'].nunique())

####################################
""" ZuCO 2.0 Normal Reading (NR) """
####################################
def extract_merge(file_dir, n=1):
    sentence_path = file_dir + f'/nr_{n}.csv'
    control_path = file_dir + f'/nr_{n}_control_questions.csv'
    df_raw = pd.read_csv(sentence_path, sep=';', encoding='utf-8', header=None,
                         names = ['paragraph_id', 'sentence_id','sentence','control'],
                         dtype={'paragraph_id':str, 'sentence_id': str, 'sentence': str, 'control': str})
    df_control = pd.read_csv(control_path, sep=';', encoding='utf-8', header=0,
                             dtype={'paragraph_id':str, 'sentence_id': str,'control_question': str, 'correct_answer':str})
    assert df_raw[df_raw['control']=='CONTROL'].shape[0] == df_control.shape[0]
    df = pd.merge(df_raw, df_control, how='left', on=['paragraph_id', 'sentence_id'])
    return df

def merge_QA(q,a):
    if pd.isna(q):
        label = np.nan
    else:
        if q.endswith('...'):
            label = q.replace('...', ' '+a)
        elif q.endswith('?'):
            label = q + ' ' + a
        else:
            raise ValueError
    return label

df22_list = []

for i in range(1,8):
    df = extract_merge(data_dir_zuco2, i)
    df22_list.append(df)
df22 = pd.concat(df22_list, ignore_index=True,)

labels=[]
for i in range(df22.shape[0]):
    label = merge_QA(df22['control_question'][i], df22['correct_answer'][i])
    labels.append(label)
df22['raw label'] = labels
df22['control'] = df22['control'].apply(lambda x: x == 'CONTROL')

df22 = df22.rename(columns={'sentence': 'raw text'})
df22 = df22.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
df22['dataset'] =  ['ZuCo2'] * df22.shape[0]
df22['task'] =  ['task2'] * df22.shape[0]

# drop unused column
df22 = df22.drop(['control'], axis = 1)

# print(df22.shape[0], df22.columns)
# print(df22['raw text'].nunique())
# print(df22['raw text'].value_counts())

###########################################
""" ZuCO 2.0 Task-specific Reading (NR) """
###########################################
def extract_task3(file_dir, n=1):
    file_path = file_dir + f'/tsr_{n}.csv'
    df_raw = pd.read_csv(file_path, sep=';', encoding='utf-8', header=None,
                         names = ['paragraph_id', 'sentence_id', 'sentence', 'label'],
                         dtype={'paragraph_id':str, 'sentence_id': str, 'sentence': str, 'label': str})
    df = df_raw.rename(columns={'sentence': 'raw text', 
                                'label': 'raw label'})
    df = df.reindex(columns=['raw text', 'dataset', 'task', 'control', 'raw label',])
    df['control'] = df['raw label'].apply(assign_control_with_label)
    unique_labels = df['raw label'].unique().tolist()
    unique_labels.remove('CONTROL')
    assert len(unique_labels) == 1
    df['raw label'] =  unique_labels * df.shape[0]
    df['dataset'] =  ['ZuCo2'] * df.shape[0]
    df['task'] =  ['task3'] * df.shape[0]
    return df

def assign_control_with_label(label):
    assert label in ['AWARD', 'EDUCATION', 'EMPLOYER', 
                   'FOUNDER', 'JOB_TITLE', 'NATIONALITY', 
                   'POLITICAL_AFFILIATION', 'VISITED', 'WIFE',
                   'CONTROL']
    return True if label == 'CONTROL' else False

df23_list = []
for i in range(1,8):
    df = extract_task3(data_dir_zuco2,i)
    df23_list.append(df)
df23 = pd.concat(df23_list, ignore_index=True,)

# drop unused column
df23 = df23.drop(['control'], axis = 1)

# print(df23.shape[0], df23.columns)
# print(df23['raw text'].nunique())

#########################
""" Concat dataframes """
#########################
df = pd.concat([df11, df12, df13, df22, df23], ignore_index = True,)
# print(df.shape, df.columns)

####################
""" Revise typo """
####################
typobook = {"emp11111ty":   "empty",
            "film.1":       "film.",
            "–":            "-",
            "’s":           "'s",
            "�s":           "'s",
            "`s":           "'s",
            "Maria":        "Marić",
            "1Universidad": "Universidad",
            "1902—19":      "1902 - 19",
            "Wuerttemberg": "Württemberg",
            "long -time":   "long-time",
            "Jose":         "José",
            "Bucher":       "Bôcher",
            "1839 ? May":   "1839 - May",
            "G�n�ration":  "Generation",
            "Bragança":     "Bragana",
            "1837?October": "1837 - October",
            "nVera-Ellen":  "Vera-Ellen",
            "write Ethics": "wrote Ethics",
            "Adams-Onis":   "Adams-Onís",
            "(40 km?)":     "(40 km²)",
            "(40 km˝)":     "(40 km²)",
            " (IPA: /?g?nz?b?g/) ": " ",
            '""Canes""':    '"Canes"',

            }

def revise_typo(text):
    # the typo book 
    book = typobook
    for src, tgt in book.items():
        if src in text:
            text = text.replace(src, tgt)
    return text

df['input text'] = df['raw text'].apply(revise_typo)

# print(df.columns)
# print(df['raw text'].nunique(), df['input text'].nunique())

#########################
""" Assign Unique IDs """
#########################
uids, unique_texts = pd.factorize(df['input text'])
df['text uid'] = uids.tolist()

#######################
""" Save to pickle """
#######################
# save dataframe
pd.to_pickle(df, tmp_path + '/zuco_label_input_text.df')
df.to_csv(tmp_path + '/zuco_label_input_text.csv')
# debug message
print(f"[INFO] Processed zuco labels saved to {tmp_path + '/zuco_label_input_text.df'}")