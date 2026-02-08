"""
This is the datamodule for SEMKEY_E2E training
"""

import os
import torch
import numpy as np
import pandas as pd
import lightning as pl
from typing import Literal
from torch.utils.data import Dataset, DataLoader

from data.datamodule import GLIMSampler
from preprocess.signal_process.functions import spectral_whitening, robust_normalize_padded

# Macros for signal preprocessing
# Please configure these in command line
SPECTRAL_WHITENING = True
ROBUST_NORMALIZE = True

# Seeding macro (set to None to use split of the dataset)
SEED = None

# MTV variant keys
VARIANT_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)',
    'semantic clarity (v0)', 'semantic clarity (v1)',
    'syntax simplification (v0)', 'syntax simplification (v1)',
    'naive rewritten', 'naive simplified'
]


class E2EDataModule(pl.LightningDataModule):
    """DataModule for E2E training that loads raw EEG and labels from separate dataframes."""

    def __init__(self,
                 eeg_data_path: os.PathLike,
                 labels_data_path: os.PathLike,
                 use_mtv: bool = False,
                 spectral_whitening: bool = SPECTRAL_WHITENING,
                 robust_normalize: bool = ROBUST_NORMALIZE,
                 batch_size: int = 24,
                 num_workers: int = 0,
                 seed: int = SEED,):
        super().__init__()
        assert os.path.exists(eeg_data_path), f"EEG data path not found: {eeg_data_path}"
        assert os.path.exists(labels_data_path), f"Labels data path not found: {labels_data_path}"

        self.eeg_data_path = eeg_data_path
        self.labels_data_path = labels_data_path
        self.use_mtv = use_mtv
        self.spectral_whitening = spectral_whitening
        self.robust_normalize = robust_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str) -> None:
        print(f'[E2EDataModule] Loading data...')

        # Load EEG dataframe (has raw EEG, variants)
        # Process Whiteing and Normalization
        # Let's cache it
        # Again BAD BAD code, but might work
        import pathlib
        if self.spectral_whitening and self.robust_normalize:
            path = pathlib.Path(self.eeg_data_path)
            cache_path = path.with_stem(path.stem + '_whiten_norm')
            if cache_path.is_file():
                eeg_df = pd.read_pickle(cache_path)
                print(f"[DEBUG] Cache loaded {cache_path}")
            else:
                # We generate the cached file
                eeg_df : pd.DataFrame = pd.read_pickle(self.eeg_data_path)
                eeg_df.loc[:, 'eeg'] = eeg_df['eeg'].apply(lambda x: robust_normalize_padded(spectral_whitening(x)))
                # Cache it
                eeg_df.to_pickle(cache_path)
        elif self.spectral_whitening:
            path = pathlib.Path(self.eeg_data_path)
            cache_path = path.with_stem(path.stem + '_whiten')
            if cache_path.is_file():
                eeg_df = pd.read_pickle(cache_path)
                print(f"[DEBUG] Cache loaded {cache_path}")
            else:
                # We generate the cached file
                eeg_df : pd.DataFrame = pd.read_pickle(self.eeg_data_path)
                eeg_df.loc[:, 'eeg'] = eeg_df['eeg'].apply(spectral_whitening)
                # Cache it
                eeg_df.to_pickle(cache_path)
        elif self.robust_normalize:
            path = pathlib.Path(self.eeg_data_path)
            cache_path = path.with_stem(path.stem + '_norm')
            if cache_path.is_file():
                eeg_df = pd.read_pickle(cache_path)
                print(f"[DEBUG] Cache loaded {cache_path}")
            else:
                # We generate the cached file
                eeg_df : pd.DataFrame = pd.read_pickle(self.eeg_data_path)
                eeg_df.loc[:, 'eeg'] = eeg_df['eeg'].apply(robust_normalize_padded)
                # Cache it
                eeg_df.to_pickle(cache_path)
        else:
            eeg_df = pd.read_pickle(self.eeg_data_path)
        print(f'  Loaded EEG data: {len(eeg_df)} samples')

        # Load phase if seed is provided
        if self.seed is not None:
            import pathlib
            path = pathlib.Path(self.eeg_data_path)
            cache_path = path.with_stem(path.stem + f"_phase_{self.seed}")
            # Check if cache exists
            if cache_path.is_file():
                # Load that cached phase df
                phase: pd.DataFrame = pd.read_pickle(cache_path)
                # Assert
                assert len(eeg_df['phase']) == len(phase['phase']), "[ERROR] Cached phase file has different length"
                # Assign phase column
                eeg_df['phase'] = phase['phase'].values
            else:
                raise FileNotFoundError(f"The required cache file '{cache_path}' was not found.")

        # Load labels dataframe (has labels from Stage 1)
        labels_df = pd.read_pickle(self.labels_data_path)
        print(f'  Loaded labels data: {len(labels_df)} samples')

        # Dataframes are aligned by index - replace label columns
        df = eeg_df.copy()
        df['sentiment label'] = labels_df['sentiment label'].values
        df['topic_label'] = labels_df['topic_label'].values
        df['length'] = labels_df['length'].values
        df['surprisal'] = labels_df['surprisal'].values
        print(f'  Replaced labels from zuco2best')

        # Check for required columns
        required_cols = ['eeg', 'mask', 'text uid', 'input text', 'sentiment label',
                        'topic_label', 'length', 'surprisal', 'task', 'dataset', 'subject', 'phase']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if stage == "fit":
            train_df = df[df['phase'] == 'train'].copy()
            val_df = df[df['phase'] == 'val'].copy()

            print(f'  Train samples: {len(train_df)}')
            print(f'  Val samples: {len(val_df)}')

            self.train_set = E2EDataset(train_df, 'train', self.use_mtv, VARIANT_KEYS)
            self.val_set = E2EDataset(val_df, 'val', False, [])

        elif stage == "test":
            test_df = df[df['phase'] == 'test'].copy()
            print(f'  Test samples: {len(test_df)}')
            self.test_set = E2EDataset(test_df, 'test', False, [])

        print(f'[E2EDataModule] Setup complete!')

    def train_dataloader(self):
        train_sampler = GLIMSampler(
            self.train_set,
            self.train_set.data['text uid'],
            'train',
            self.batch_size
        )
        return DataLoader(
            self.train_set,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        val_sampler = GLIMSampler(
            self.val_set,
            self.val_set.data['text uid'],
            'val',
            self.batch_size
        )
        return DataLoader(
            self.val_set,
            batch_sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        test_sampler = GLIMSampler(
            self.test_set,
            self.test_set.data['text uid'],
            'test',
            self.batch_size
        )
        return DataLoader(
            self.test_set,
            batch_sampler=test_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )


class E2EDataset(Dataset):
    """Dataset for E2E training that returns raw EEG and labels."""

    def __init__(self,
                 df: pd.DataFrame,
                 phase: Literal['train', 'val', 'test'],
                 use_mtv: bool,
                 variant_keys: list):
        self.phase = phase

        # Apply MTV augmentation for training only
        if use_mtv and phase == 'train' and len(variant_keys) > 0:
            print(f'  Applying MTV augmentation to training set...')
            print(f'    Original size: {len(df)}')

            # Check if variant columns exist
            missing_variants = [k for k in variant_keys if k not in df.columns]
            if missing_variants:
                print(f'    WARNING: Missing variant columns: {missing_variants}')
                variant_keys = [k for k in variant_keys if k in df.columns]

            if len(variant_keys) > 0:
                # Unpivot: create a row for each variant
                id_cols = [c for c in df.columns if c not in variant_keys]
                df = df.melt(
                    id_vars=id_cols,
                    value_vars=variant_keys,
                    value_name='target text'
                ).drop(columns=['variable'])

                print(f'    Augmented size: {len(df)} ({len(variant_keys)}x)')
            else:
                print(f'    No valid variant columns found, skipping MTV')
                df['target text'] = df['input text']
        else:
            # Use input text as target text
            df['target text'] = df['input text']

        # Extract data
        self.data = {
            'eeg': df['eeg'].tolist(),
            'mask': df['mask'].tolist(),
            'text uid': df['text uid'].tolist(),
            'input text': df['input text'].tolist(),
            'target text': df['target text'].tolist(),
            'sentiment label': df['sentiment label'].tolist(),
            'topic_label': df['topic_label'].tolist(),
            'length': df['length'].tolist(),
            'surprisal': df['surprisal'].tolist(),
            'task': ['<NR>' if t != 'task3' else '<TSR>' for t in df['task'].tolist()],
            'dataset': df['dataset'].tolist(),
            'subject': df['subject'].tolist()
        }

    def __len__(self):
        return len(self.data['eeg'])

    def __getitem__(self, idx):
        return {
            'eeg': torch.from_numpy(self.data['eeg'][idx]).float(),
            'mask': torch.from_numpy(self.data['mask'][idx]),
            'prompt': (self.data['task'][idx], self.data['dataset'][idx], self.data['subject'][idx]),
            'text uid': self.data['text uid'][idx],
            'input text': self.data['input text'][idx],
            'target text': self.data['target text'][idx],
            'sentiment label': self.data['sentiment label'][idx],
            'topic_label': self.data['topic_label'][idx],
            'length': self.data['length'][idx],
            'surprisal': self.data['surprisal'][idx]
        }
