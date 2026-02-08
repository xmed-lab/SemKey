"""
This is the datamodule for SEMKEY stage 1 training
-> Since this piece of code is derived from GLIM datamodule
-> We highly value GLIM's work, so, let's keep their name here
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
import warnings
import lightning as pl
import torch.distributed as dist
from typing import Literal, Iterator, Union
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from preprocess.signal_process.functions import spectral_whitening, robust_normalize_padded

"""
WARNING: The newly added macros (below) are for separate testing
-> Most of them are configurable using command line arguments
-> Split ratio (re-splitting) can ONLY be set here
"""
# Constants
SBERT_EMBEDDING_DIM = 768  # Dimension of SBERT (all-mpnet-base-v2) embeddings
MAX_KEYWORDS = 3  # Number of top keywords to extract per sentence

# Macros for signal preprocessing
SPECTRAL_WHITENING = True
ROBUST_NORMALIZE = True

# classification label
CLS_LABEL = 'topic_label'

# Seeding macro (set to None to use split of the dataset)
SEED = None
# (Split size - hard coded)
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% of total data

class GLIMDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path: os.PathLike,
                 embeddings_path: os.PathLike = None,
                 eval_noise_input: bool = False,
                 bsz_train = 64,
                 bsz_val = 24,
                 bsz_test = 24,
                 test_set_key: Literal['test', 'train', 'val'] = 'test',
                 num_workers: int = 0,
                 use_weighted_sampler: bool = False,
                 classification_label_key: str = None,
                 classification_label_keys: list = None,
                 regression_label_keys: list = None,
                 use_zuco1_only: bool = False,
                 use_spectral_whitening: bool = SPECTRAL_WHITENING,
                 use_robust_normalize: bool = ROBUST_NORMALIZE,
                 seed: int = SEED,
                 ):
        super().__init__()
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.eval_noise_input = eval_noise_input
        self.bsz_train = bsz_train
        self.bsz_val = bsz_val
        self.bsz_test = bsz_test
        self.test_set_key = test_set_key
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        self.use_zuco1_only = use_zuco1_only
        self.use_spectral_whitening = use_spectral_whitening
        self.use_robust_normalize = use_robust_normalize
        self.seed = seed # Set to None to use default split

        # Handle both single and multi-task modes
        if classification_label_keys is not None:
            self.classification_label_keys = classification_label_keys
        elif classification_label_key is not None:
            self.classification_label_keys = [classification_label_key]
        else:
            self.classification_label_keys = [CLS_LABEL]

        # For backward compatibility
        self.classification_label_key = self.classification_label_keys[0]

        self.regression_label_keys = regression_label_keys or []

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def shuffle_dataframe(self, seed: int, df: pd.DataFrame) -> pd.DataFrame:
        # Check if we have it cached
        import pathlib
        path = pathlib.Path(self.data_path)
        cache_path = path.with_stem(path.stem + f"_phase_{seed}")

        # Create and cache new split
        if (not cache_path.is_file()):
            # Import package for the split
            from sklearn.model_selection import train_test_split
            # Split by text uid to ensure no data leakage
            unique_text_uids = df['text uid'].unique()
            train_uids, test_uids = train_test_split(unique_text_uids, test_size=TEST_SIZE, random_state=seed)
            # Calculate correct validation split size from remaining training data
            val_split_ratio = VAL_SIZE / (1 - TEST_SIZE)
            train_uids, val_uids = train_test_split(train_uids, test_size=val_split_ratio, random_state=seed)
            def assign_phase(text_uid):
                if text_uid in train_uids:
                    return 'train'
                elif text_uid in val_uids:
                    return 'val'
                else:
                    return 'test'
            # Override existing phase
            df.loc[ : , 'phase'] = df['text uid'].apply(assign_phase)
            # Let's cache it (only save phase)
            df[['phase']].to_pickle(cache_path)
            # Reture
            return df
        else:
            # Load that cached phase df
            phase: pd.DataFrame = pd.read_pickle(cache_path)
            # Assert
            assert len(df['phase']) == len(phase['phase']), "[ERROR] Cached phase file has different length"
            # Assign phase column
            df['phase'] = phase['phase'].values
            # return
            return df

    def setup(self, stage: str) -> None:
        try:
            local_rank = os.environ["LOCAL_RANK"]
        except KeyError:
            local_rank = '0'
        print(f'[Rank {local_rank}][{self.__class__.__name__}] running `setup()`...', end='\n')

        # Process Whiteing and Normalization
        # Let's cache it
        # Again BAD BAD code, but might work
        import pathlib
        if self.use_spectral_whitening and self.use_robust_normalize:
            path = pathlib.Path(self.data_path)
            cache_path = path.with_stem(path.stem + '_whiten_norm')
            if cache_path.is_file():
                df = pd.read_pickle(cache_path)
            else:
                # We generate the cached file
                df : pd.DataFrame = pd.read_pickle(self.data_path)
                df.loc[:, 'eeg'] = df['eeg'].apply(lambda x: robust_normalize_padded(spectral_whitening(x)))
                # Cache it
                df.to_pickle(cache_path)
        elif self.use_spectral_whitening:
            path = pathlib.Path(self.data_path)
            cache_path = path.with_stem(path.stem + '_whiten')
            if cache_path.is_file():
                df = pd.read_pickle(cache_path)
            else:
                # We generate the cached file
                df : pd.DataFrame = pd.read_pickle(self.data_path)
                df.loc[:, 'eeg'] = df['eeg'].apply(spectral_whitening)
                # Cache it
                df.to_pickle(cache_path)
        elif self.use_robust_normalize:
            path = pathlib.Path(self.data_path)
            cache_path = path.with_stem(path.stem + '_norm')
            if cache_path.is_file():
                df = pd.read_pickle(cache_path)
            else:
                # We generate the cached file
                df : pd.DataFrame = pd.read_pickle(self.data_path)
                df.loc[:, 'eeg'] = df['eeg'].apply(robust_normalize_padded)
                # Cache it
                df.to_pickle(cache_path)
        else:
            df = pd.read_pickle(self.data_path)

        # Shuffle dataset and cache if needed
        # Use a separate function for NICER coding
        # Maybe not NICE at all ... :-(
        if self.seed is not None:
            df = self.shuffle_dataframe(self.seed, df)

        if self.use_zuco1_only:
            df = df[df['dataset'] == 'ZuCo1']

        # Load embeddings if path is provided
        embeddings_dict = None
        if self.embeddings_path is not None and os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                embeddings_dict = pickle.load(f)
            print(f'[Rank {local_rank}] Loaded embeddings from {self.embeddings_path}')
        
        if stage == "fit":
            self.train_set = ZuCoDataset(df, 'train', embeddings_dict=embeddings_dict, classification_label_keys=self.classification_label_keys, regression_label_keys=self.regression_label_keys)
            self.val_set = ZuCoDataset(df, 'val', embeddings_dict=embeddings_dict, eval_noise_input=self.eval_noise_input, classification_label_keys=self.classification_label_keys, regression_label_keys=self.regression_label_keys)
            self.n_target_text = self.val_set.n_target_text
        elif stage == "test":
            self.test_set = ZuCoDataset(df, 'test', embeddings_dict=embeddings_dict, eval_noise_input=self.eval_noise_input, classification_label_keys=self.classification_label_keys, regression_label_keys=self.regression_label_keys)
            self.n_target_text = self.test_set.n_target_text
        print(f'[Rank {local_rank}][{self.__class__.__name__}] running `setup()`...Done!','\U0001F60B'*3)
            
    def train_dataloader(self):
        if self.use_weighted_sampler:
            # Create weighted sampler based on classification labels
            train_sampler = WeightedGLIMSampler(
                self.train_set,
                self.train_set.data['text uid'],
                self.train_set.data[self.classification_label_key],
                'train',
                self.bsz_train
            )
        else:
            train_sampler = GLIMSampler(
                self.train_set,
                self.train_set.data['text uid'],
                'train',
                self.bsz_train
            )
        train_loader = DataLoader(self.train_set,
                                  batch_sampler=train_sampler,
                                  num_workers = self.num_workers,
                                  pin_memory=True,
                                  )
        return train_loader

    def val_dataloader(self):
        val_sampler = GLIMSampler(self.val_set, self.val_set.data['text uid'], 
                                  'val', self.bsz_val)
        val_loader = DataLoader(self.val_set,
                                batch_sampler = val_sampler,
                                num_workers = self.num_workers,
                                pin_memory=True,
                                )
        return val_loader
    
    def test_dataloader(self):
        test_sampler = GLIMSampler(self.test_set, self.test_set.data['text uid'], 
                                   'test', self.bsz_test)
        test_loader = DataLoader(self.test_set,
                                 batch_sampler = test_sampler,
                                 num_workers = self.num_workers,
                                 pin_memory=True,
                                 )
        return test_loader


class GLIMSampler(DistributedSampler):
    '''
    A batch sampler for train/val/test GLIM on `ZuCo1` + `ZuCo2`.  
    It samples batches by the `text` rather than `eeg-text pair` to make sure the `clip loss` works properly
    '''
    def __init__(self,
                 dataset: Dataset,
                 identifiers: list,
                 phase: Literal['train', 'val', 'test'],
                 batch_size: int,
                 num_replicas = None,
                 rank = None,
                 drop_last = None,
                 ) -> None:
        if (num_replicas is None) and (not dist.is_initialized()):
            self.dataset = dataset
            self.num_replicas = 1
            self.rank = 0
            self.epoch = 0
            self.seed = 0
        else:
            super().__init__(dataset, num_replicas=num_replicas, rank=rank)
            # set 4 attributes inside: self.dataset, self.num_replicas, self.rank, self.epoch, self.seed
            del self.num_samples
            del self.total_size

        # Use provided drop_last if given, otherwise set based on phase
        if drop_last is not None:
            self.drop_last = drop_last
            self.shuffle = (True if phase == 'train' else False)
        else:
            self.shuffle, self.drop_last = (True, True) if phase == 'train' else (False, True)
        # NOTE: drop_last is inevitable, see `sample_batches()`
        self.phase = phase
        self.batch_size = batch_size
        self.identifiers = torch.tensor(identifiers)
        
        # uni_text_uids, counts = torch.unique(self.text_ids, return_counts=True, dim=0)
        # n_uni_text = len(uni_text_uids)
        # assert n_uni_text // (self.num_replicas * self.batch_size) > 0
        self.n_batches_per_device = self.estimate_len()
        self.n_batches = self.n_batches_per_device * self.num_replicas

    def __len__(self) -> int:
        return self.n_batches_per_device

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        batches, _ = self.sample_batches(indices,
                                        identifiers = self.identifiers[indices], 
                                        batch_size = self.batch_size,
                                        )
        if len(batches) >= self.n_batches:
            batches = batches[:self.n_batches]
        else:
            padding_size = self.n_batches - len(batches)
            if padding_size > 3:
                print('😅😅😅 [padding_size>3]!!! ', f'expect {self.n_batches} batches but only got {len(batches)}...')
                print('epoch:             ',f'{self.epoch}')
                print('phase:             ',f'{self.phase}')
                print('batch_size:        ',f'{self.batch_size}')
            batches += batches[:padding_size]
        sub_batches = batches[self.rank : self.n_batches : self.num_replicas]

        for batch in sub_batches:
            yield batch

    def estimate_len(self, k=10):
        if self.shuffle:
            batch_nums = []
            for i in range(k):
                g = torch.Generator()
                g.manual_seed(self.seed + i)  # for reproducibility
                indices = torch.randperm(len(self.dataset), generator=g)
                batches, _ = self.sample_batches(indices,
                                                identifiers = self.identifiers[indices], 
                                                batch_size = self.batch_size,
                                                )
                batch_nums.append(len(batches))
            batch_num = min(batch_nums)
        else:
            indices = torch.arange(len(self.dataset))
            batches, _ = self.sample_batches(indices,
                                            identifiers = self.identifiers[indices], 
                                            batch_size = self.batch_size,
                                            )
            batch_num = len(batches)
        estimated_len = batch_num // self.num_replicas
        return estimated_len
    
    @torch.no_grad()
    def sample_batches(self, indices, identifiers, batch_size, exhaust = False):
        '''
        Sampling batches by `identifiers` rather than `indices` of samples makes sure that 
        all samples within a batch have distinct identifiers.

        Inputs:
            `indices`:      torch.tensor, (n), sample indices ranging from 0 to n-1
            `indentifiers`: torch.tensor, (n), `text uids` with the same order coressponding to `indices`
        '''
        # Group samples by `task id`
        ids_idents = torch.stack([indices, identifiers],dim=1)
        batches = []
        
        unused_ids_idents = ids_idents
        n_uni_idents = torch.unique(ids_idents[:,1]).shape[0]
        # print('n_uni_idents:      ',f'{n_uni_idents}')
        while n_uni_idents >= batch_size:
            valid_batches, unused_ids_idents = self.non_overlapping_sample(unused_ids_idents, self.batch_size)
            batches.extend(valid_batches)
            if len(unused_ids_idents) == 0:
                n_uni_idents = 0
                break
            n_uni_idents = torch.unique(unused_ids_idents[:,1]).shape[0]  
        # batches: list[Tensor(bs, 2)]
        batches_ids = [batch[:,0].int().tolist() for batch in batches] # list[list[int]]

        exhausted_batches = []
        if exhaust and len(unused_ids_idents) != 0: 
            while n_uni_idents > 0:
                valid_batches, unused_ids_idents = self.non_overlapping_sample(unused_ids_idents, n_uni_idents)
                exhausted_batches.extend(valid_batches)
                if len(unused_ids_idents) == 0:
                    break
                n_uni_idents = torch.unique(unused_ids_idents[:,1]).shape[0] 
            # exhausted_batches = list[Tensor(bs*, 2)]
            exhausted_batches_ids = [batch[:,0].int().tolist() for batch in exhausted_batches] # list[list[int]]
        else:
            exhausted_batches_ids = []
        return batches_ids, exhausted_batches_ids
    
    def non_overlapping_sample(self, ids_idents: torch.Tensor, 
                               batch_size: int) -> tuple[list[torch.Tensor], Union[torch.Tensor, list]]:
        valid_batches = []
        used_idents = set()
        current_batch = []
        unused_samples = []
        for idx_ident in ids_idents:
            idx, ident = idx_ident
            if ident.item() not in used_idents: # Track identifiers used in the current batch to ensure uniqueness
                used_idents.add(ident.item())
                current_batch.append(idx_ident)
            else:
                unused_samples.append(idx_ident)

            if len(current_batch) == batch_size:
                valid_batches.append(torch.stack(current_batch))
                current_batch = []
                used_idents = set()
        # Include the last batch if it has fewer than batch_size elements
        if current_batch:
            unused_samples.extend(current_batch)
        try:
            unused_samples = torch.stack(unused_samples)
        except:
            unused_samples = []
        return (valid_batches,   # list[tensor(bs, 2)]
                unused_samples,  # tensor(n, 2)
                )


class WeightedGLIMSampler(GLIMSampler):
    '''
    A weighted batch sampler for train GLIM that applies class balancing based on classification labels.
    Extends GLIMSampler to maintain the text-based sampling while applying weights for class imbalance.
    '''
    def __init__(self,
                 dataset: Dataset,
                 identifiers: list,
                 classification_labels: list,
                 phase: Literal['train', 'val', 'test'],
                 batch_size: int,
                 num_replicas = None,
                 rank = None,
                 drop_last = None,
                 ) -> None:
        super().__init__(dataset, identifiers, phase, batch_size, num_replicas, rank, drop_last)
        
        # Compute sample weights based on classification labels
        self.sample_weights = self._compute_sample_weights(classification_labels)
    
    def _compute_sample_weights(self, classification_labels):
        """
        Compute sample weights for weighted random sampling to handle class imbalance.
        Uses inverse frequency weighting.
        
        Args:
            classification_labels: List of classification label strings
            
        Returns:
            Tensor of sample weights, one per sample
        """
        # Map classification labels to IDs
        label_to_id = list(set(classification_labels))
        cls_lbl_ids = torch.tensor([label_to_id.index(label) for label in classification_labels])
        
        # Count samples per class
        unique_labels, counts = torch.unique(cls_lbl_ids, return_counts=True)
        
        # Compute class weights (inverse frequency)
        total_samples = len(cls_lbl_ids)
        class_weights = total_samples / (len(unique_labels) * counts.float())
        
        # Create a mapping from label to weight
        label_to_weight = {label.item(): weight.item() for label, weight in zip(unique_labels, class_weights)}
        
        # Assign weight to each sample based on its label
        sample_weights = torch.tensor([label_to_weight[label.item()] for label in cls_lbl_ids])
        
        return sample_weights
    
    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Apply weighted sampling
            weighted_sampler = torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.dataset),
                replacement=True,
                generator=g
            )
            indices = torch.tensor(list(weighted_sampler))
        else:
            indices = torch.arange(len(self.dataset))

        batches, _ = self.sample_batches(indices,
                                        identifiers = self.identifiers[indices], 
                                        batch_size = self.batch_size,
                                        )
        if len(batches) >= self.n_batches:
            batches = batches[:self.n_batches]
        else:
            padding_size = self.n_batches - len(batches)
            if padding_size > 3:
                print('😅😅😅 [padding_size>3]!!! ', f'expect {self.n_batches} batches but only got {len(batches)}...')
                print('epoch:             ',f'{self.epoch}')
                print('phase:             ',f'{self.phase}')
                print('batch_size:        ',f'{self.batch_size}')
            batches += batches[:padding_size]
        sub_batches = batches[self.rank : self.n_batches : self.num_replicas]

        for batch in sub_batches:
            yield batch


class ZuCoDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 phase: Literal['train', 'val', 'test'],
                 embeddings_dict: dict = None,
                 eval_noise_input: bool = False,
                 classification_label_key: str = None,
                 classification_label_keys: list = None,
                 regression_label_keys: list = None
                 ):
        
        # Handle both single and multi-task modes
        if classification_label_keys is not None:
            self.classification_label_keys = classification_label_keys
        elif classification_label_key is not None:
            self.classification_label_keys = [classification_label_key]
        else:
            self.classification_label_keys = [CLS_LABEL]

        # For backward compatibility
        self.classification_label_key = self.classification_label_keys[0]

        self.regression_label_keys = regression_label_keys or []

        # pt_target_keys = ['input text']
        pt_target_keys = ['lexical simplification (v0)', 'lexical simplification (v1)', 
                          'semantic clarity (v0)', 'semantic clarity (v1)', 
                          'syntax simplification (v0)', 'syntax simplification (v1)',
                          'naive rewritten', 'naive simplified']
        df = df[df['phase'] == phase]
        if phase == 'train':
            target_keys = pt_target_keys
            data_dicts = []
            for target_key in target_keys:
                data = self.__fetch_from_df(df, target_key, embeddings_dict)
                data_dicts.append(data)
            data = collate_fn(data_dicts)
            targets_tuple_list = [(text, ) for text in data['target text']]
        else:
            data = self.__fetch_from_df(df, "input text", embeddings_dict)
            if eval_noise_input:
                data['eeg']
            target_lists = [df[key].values.tolist() for key in pt_target_keys]
            targets_tuple_list = list(zip(*target_lists))
        data.update({"all target texts": targets_tuple_list})

        if eval_noise_input:
            n = len(data['eeg'])
            l,c = data['eeg'][0].shape
            data.pop('eeg')
            data.pop('mask')
            rng = np.random.default_rng(seed=42)
            data['eeg'] = rng.standard_normal((n,l,c), dtype=np.float32)
            data['mask'] = np.ones((n,l), dtype=np.int8)

        self.n_target_text = len(pt_target_keys)
        self.data = data
        
    def __fetch_from_df(self, df, target_key, embeddings_dict=None):
        

        input_template = "To English: <MASK>"
        raw_input_text = df['input text'].tolist()
        input_text = [input_template.replace("<MASK>", src) for src in raw_input_text]
        target_text = df[target_key].tolist()
        
        raw_t_keys = df['task'].tolist()
        t_prompts = ['<NR>' if t_key != 'task3' else '<TSR>' for t_key in raw_t_keys]
        # t_prompts = ['<NR>'] * len(raw_t_keys)
        d_prompts = df['dataset'].tolist()
        s_prompts = df['subject'].tolist()
        prompt = list(zip(t_prompts, d_prompts, s_prompts))
        text_uid = df['text uid'].values.tolist()

        # Load EEG and mask
        eeg = df['eeg'].tolist()
        mask = df['mask'].tolist()

        # Load embeddings if available
        sentence_embeddings = []
        keyword_embeddings = []
        keyword_texts = []
        if embeddings_dict is not None:
            for uid in text_uid:
                if uid in embeddings_dict:
                    sentence_embeddings.append(embeddings_dict[uid]['sentence'])
                    keyword_embeddings.append(embeddings_dict[uid]['keyword'])
                else:
                    # If embedding not found, use zeros as placeholder
                    warnings.warn(
                        f"Embedding not found for text_uid {uid}. Using zero placeholder. "
                        f"This may indicate missing data in embeddings.pickle or a mismatch in preprocessing.",
                        UserWarning
                    )
                    sentence_embeddings.append(np.zeros(SBERT_EMBEDDING_DIM, dtype=np.float32))
                    keyword_embeddings.append(np.zeros((MAX_KEYWORDS, SBERT_EMBEDDING_DIM), dtype=np.float32))

            # Get keyword texts from dataframe
            if 'keyword_1' in df.columns:
                keyword_texts = list(zip(df['keyword_1'].tolist(),
                                        df['keyword_2'].tolist(),
                                        df['keyword_3'].tolist()))

        result = {'eeg': eeg,                   # list[np.arrary], [(l, c),]
                'mask': mask,                 # list[np.arrary], [(l),], 1 for unmasked; 0 for masked
                'prompt': prompt,             # list[tuple[str]], [('task', 'dataset', 'subject')]
                'text uid': text_uid,         # list[int]
                'input text': input_text,     # str
                'target text': target_text,   # str
                'raw task key': raw_t_keys,                                 # str
                'raw input text': raw_input_text,                           # str
                }

        # Add classification labels
        for label_key in self.classification_label_keys:
            result[label_key] = df[label_key].values.tolist()

        # Add regression labels
        for label_key in self.regression_label_keys:
            result[label_key] = df[label_key].values.tolist()
        
        # Add embeddings if available
        if embeddings_dict is not None:
            result['sentence_embedding'] = sentence_embeddings  # list[np.array], [(768,)]
            result['keyword_embedding'] = keyword_embeddings    # list[np.array], [(3, 768)]
            result['keyword_text'] = keyword_texts              # list[tuple[str]], [(kw1, kw2, kw3)]
        
        return result

    def __len__(self):
        return len(self.data['eeg'])
    
    def __getitem__(self, idx):
        item = {
                'eeg': torch.from_numpy(self.data['eeg'][idx]).float(),       # tensor, float32, (*, l, c) -> MAKE sure it's of the right type (especially when mixed-precision is enabled)
                'mask': torch.from_numpy(self.data['mask'][idx]),     # tensor, int8, (*, l), 1 for unmasked; 0 for masked
                'prompt': self.data['prompt'][idx],             # tuple[str], [('task', 'dataset', 'subject')]
                'text uid': self.data['text uid'][idx],         # int
                'input text': self.data['input text'][idx],     # str
                'target text': self.data['target text'][idx],   # str
                'raw task key': self.data['raw task key'][idx],                                  # str
                'raw input text': self.data['raw input text'][idx],                              # str
                'all target texts': self.data['all target texts'][idx],                          # tuple(str)
                }

        # Add classification labels
        for label_key in self.classification_label_keys:
            item[label_key] = self.data[label_key][idx]

        # Add regression labels
        for label_key in self.regression_label_keys:
            item[label_key] = self.data[label_key][idx]

        # Add embeddings if available
        if 'sentence_embedding' in self.data:
            item['sentence_embedding'] = torch.from_numpy(self.data['sentence_embedding'][idx])  # tensor, float32, (768,)
        if 'keyword_embedding' in self.data:
            item['keyword_embedding'] = torch.from_numpy(self.data['keyword_embedding'][idx])    # tensor, float32, (3, 768)
        if 'keyword_text' in self.data:
            item['keyword_text'] = self.data['keyword_text'][idx]  # tuple[str], (kw1, kw2, kw3)

        return item


def collate_fn(batch_list: list[dict]) -> dict:
    collated_batch = {}
    for k, v in batch_list[0].items():
        if isinstance(v, torch.Tensor):
            collated_batch[k] = torch.cat([batch[k] for batch in batch_list])
        else:
            collated_batch[k] = []
            for batch in batch_list:
                collated_batch[k].extend(batch[k])
    return collated_batch
