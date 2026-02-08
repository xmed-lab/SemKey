import os
import scipy
import h5py
import numpy as np
import pandas as pd
from glob import glob
import rich.progress as rp
from typing import Literal

def mat2df_zuco(dataset_name: Literal['ZuCo1','ZuCo2'],
                eeg_src_dir: os.PathLike, 
                task_dir_names: list[str],
                task_keys: list[str],
                subject_keys: list[str],
                n_sentences: list[str], # see zuco paper
                src_sample_rate = 500,
                tgt_sample_rate = 128, 
                tgt_max_len = 1280, 
                tgt_width = 128,
                ) -> pd.DataFrame:
    
    n_subjects = len(subject_keys)
    n_records_expected = [x * n_subjects for x in n_sentences] 
    assert tgt_sample_rate <= src_sample_rate
    
    with rp.Progress(rp.SpinnerColumn(),
            rp.TextColumn("[progress.description]{task.description}"),
            rp.BarColumn(),
            rp.TaskProgressColumn(),
            "•", rp.TextColumn(("Total: {task.total} Recorded: {task.fields[n_recorded]} "
                                "Dropped: {task.fields[dropped_records]} ({task.fields[drop_rate]:.2f}%)")),
            "•", rp.TimeElapsedColumn()) as prep:
        records = []
        dropped_lens = []
        # unmatched_sentences = []
        for i, task_dir_name in enumerate(task_dir_names): # iterate over 3 tasks

            mat_dir = eeg_src_dir + f'/{task_dir_name}/Matlab_files'
            mat_paths = sorted(glob(mat_dir + '/*.mat', recursive=False))
            assert len(mat_paths) == n_subjects, f'{task_dir_name}:We have 12 subjects for each task!'
            n_recorded = 0
            dropped_records = 0
            drop_rate = 0
            task_key = task_keys[i]
            dataset_key = dataset_name
            task_proc = prep.add_task(f'Proc {task_key}...', 
                                      total= n_records_expected[i], 
                                      n_recorded = n_recorded,
                                      dropped_records=dropped_records, 
                                      drop_rate=drop_rate)

            for mat_path in mat_paths: # iterate over 12 subjects
                subject_key = os.path.basename(mat_path).split('_')[0].replace('results','').strip()
                assert subject_key in subject_keys
                if dataset_name == 'ZuCo1':
                    task_records = scipy.io.loadmat(mat_path, squeeze_me=True, 
                                                    struct_as_record=False)['sentenceData']
                    n = len(task_records)
                elif dataset_name == 'ZuCo2':
                    mat = h5py.File(mat_path, 'r')
                    n = len(mat['sentenceData']['rawData'])
                assert n_sentences[i] == n, \
                    f'the actual num of sentences ({n}) does not match the expectation ({n_sentences[i]})'

                for j in range(n): 
                    if dataset_name == 'ZuCo1':
                        eeg_raw = task_records[j].rawData  # the raw sentence-level EEG time-series, 
                        text_raw = task_records[j].content
                    elif dataset_name == 'ZuCo2':
                        eeg_raw = mat[mat['sentenceData']['rawData'][j][0]][:].T.astype(np.float32)
                        text_raw = ''.join(chr(int(k)) for k in mat[mat['sentenceData']['content'][j][0]][:].squeeze())
                    
                    # exclude nan/inf eeg samples
                    if not np.all(np.isfinite(eeg_raw)):  
                        dropped_records += 1
                        continue
                        
                    assert eeg_raw[-1].any() == False 
                    # NOTE: the last channel is all empty!!!
                    # why has this never been mentioned before? even in the original paper/repo.
                    eeg104 = eeg_raw[:-1, :]  # (104, x)

                    width, len_raw = eeg104.shape
                    if len_raw < 0.5*src_sample_rate or len_raw > 10*src_sample_rate: # (0.5s, 12s) at 500Hz 
                        dropped_records += 1
                        dropped_lens.append(len_raw)
                        continue

                    len_new = int(len_raw * tgt_sample_rate / src_sample_rate)
                    eeg = scipy.signal.resample(eeg104, len_new, axis=1)  # dtype=float32
                    eeg = np.pad(eeg, ((0, tgt_width - width), (0, tgt_max_len - len_new)), 
                                 'constant', constant_values=0)
                    mask = np.zeros(tgt_max_len, dtype=np.int8) 
                    mask[:len_new] = 1  # 1 for `not masked`, 0 for `masked`
                    records.append({
                                    'eeg': eeg.T, 
                                    'mask': mask,
                                    'text': text_raw,
                                    'dataset': dataset_key,
                                    'task': task_key,
                                    'subject': subject_key
                                    })
                    n_recorded += 1
                        
                    drop_rate = (dropped_records / (n_recorded + dropped_records)) * 100
                    prep.update(task_proc, advance=1, n_recorded = n_recorded, 
                                dropped_records=dropped_records, drop_rate=drop_rate)
    print(f'Done! {len(records)} / {sum(n_records_expected)} are recorded!')      
    print(f'{len(dropped_lens)} / {sum(n_records_expected)-len(records)} are dropped due to the length!') 
    # print(drop_lens)             
    df = pd.DataFrame(records) 
    return df
