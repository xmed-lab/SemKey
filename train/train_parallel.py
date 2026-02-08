"""
Training script for SEMKEY_PARALLEL model.
"""

import os
import sys
import atexit
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.distributed as dist
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)

from data.datamodule import GLIMDataModule
from model.semkey_parallel import SEMKEY_PARALLEL

torch.set_float32_matmul_precision('medium')

def is_main_process():
    """Check if this is the main process (rank 0) in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Check environment variables for distributed training
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    return rank == 0


class TeeLogger: 
    """Logger that writes to both stdout/stderr and a file. 
    
    Automatically closes the file on program exit.
    """
    _instances = []
    
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'w')
        TeeLogger._instances.append(self)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        """Return whether the terminal stream is a TTY."""
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()
    
    def close(self):
        if self.log and not self.log.closed:
            self.log.close()
    
    @classmethod
    def close_all(cls):
        for instance in cls._instances:
            instance.close()


# Register cleanup function to close log files on exit
atexit.register(TeeLogger.close_all)

# Configuration constants
SUPPORTED_TEXT_MODELS = [
    'google/flan-t5-xl',
    'google/flan-t5-large',
    'facebook/bart-large-cnn',
    'jbochi/madlad400-3b-mt'
]

# Model defaults
DEFAULT_INPUT_EEG_LEN = 1280
DEFAULT_HIDDEN_EEG_LEN = 96
DEFAULT_INPUT_TEXT_LEN = 96
DEFAULT_INPUT_DIM = 128


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SEMKEY_PARALLEL model (multi-task: 2 classification + 2 regression)'
    )

    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/zuco_preprocessed_dataframe/zuco_merged.df',
        help='Path to the merged dataset pickle file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for resuming training or fine-tuning (optional)'
    )

    # Multi-task classification arguments
    parser.add_argument(
        '--sentiment_labels',
        type=str,
        nargs='+',
        default=['non_neutral', 'neutral'],
        help='Sentiment classification labels'
    )
    parser.add_argument(
        '--topic_labels',
        type=str,
        nargs='+',
        default=['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment'],
        help='Topic classification labels'
    )
    
    # GLIM encoder arguments (yes this part is GLIM architecture)
    parser.add_argument(
        '--text_model',
        type=str,
        default='google/flan-t5-large',
        choices=SUPPORTED_TEXT_MODELS,
        help='Pre-trained text model to use'
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default=None,
        help='Directory to cache downloaded model weights (avoids re-downloading on AFS restart)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension size for GLIM'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=1024,
        help='Embedding dimension size for GLIM'
    )
    parser.add_argument(
        '--n_in_blocks',
        type=int,
        default=6,
        help='Number of encoder blocks'
    )
    parser.add_argument(
        '--n_out_blocks',
        type=int,
        default=6,
        help='Number of decoder blocks'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--encoder_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for encoder'
    )
    parser.add_argument(
        '--hidden_eeg_len',
        type=int,
        default=DEFAULT_HIDDEN_EEG_LEN,
        help='Output sequence length for EEG encoder (number of learnable queries)'
    )
    parser.add_argument(
        '--do_not_use_prompt',
        action='store_true',
        help='Whether or not to use prompt embeddings'
    )
    
    # MLP classifier arguments
    parser.add_argument(
        '--mlp_hidden_dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions for MLP classifier'
    )
    parser.add_argument(
        '--mlp_dropout',
        type=float,
        default=0.3,
        help='Dropout rate for MLP classifier'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Freeze encoder weights (only train MLP)'
    )
    
    # Loss weight arguments
    parser.add_argument(
        '--clip_loss_weight',
        type=float,
        default=0.5,
        help='Weight for contrastive (CLIP) loss'
    )
    parser.add_argument(
        '--lm_loss_weight',
        type=float,
        default=0.0,
        help='Weight for language model loss'
    )
    parser.add_argument(
        '--commitment_loss_weight',
        type=float,
        default=0.0,
        help='Weight for commitment loss'
    )
    parser.add_argument(
        '--sentiment_loss_weight',
        type=float,
        default=0.25,
        help='Weight for sentiment classification loss'
    )
    parser.add_argument(
        '--topic_loss_weight',
        type=float,
        default=0.25,
        help='Weight for topic classification loss'
    )
    parser.add_argument(
        '--length_loss_weight',
        type=float,
        default=0.25,
        help='Weight for length regression loss'
    )
    parser.add_argument(
        '--surprisal_loss_weight',
        type=float,
        default=0.25,
        help='Weight for surprisal regression loss'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=24,
        help='Training batch size'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=24,
        help='Validation batch size'
    )
    parser.add_argument(
        '--spectral_whitening',
        action='store_true',
        default=True,
        help='Apply spectral whitening to EEG signals'
    )
    parser.add_argument(
        '--no_spectral_whitening',
        action='store_false',
        dest='spectral_whitening',
        help='Disable spectral whitening'
    )
    parser.add_argument(
        '--robust_normalize',
        action='store_true',
        default=True,
        help='Apply robust normalization to EEG signals'
    )
    parser.add_argument(
        '--no_robust_normalize',
        action='store_false',
        dest='robust_normalize',
        help='Disable robust normalization'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs'
    )
    
    # Optimizer arguments (Cosine LR with warmup)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Maximum learning rate'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help='Minimum learning rate for cosine schedule'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=0,
        help='Number of warmup epochs for learning rate scheduler'
    )
    
    # Logging and checkpoint arguments
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Base directory for logs (TensorBoard logs and checkpoints will be saved here)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='semkey_parallel',
        help='Name of the experiment'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'tpu'],
        help='Accelerator type'
    )
    parser.add_argument(
        '--device',
        type=int,
        nargs='+',
        default=[0],
        help='GPU Index [list]'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='32',
        choices=['32', '16-mixed', 'bf16-mixed'],
        help='Training precision'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=['auto', 'ddp', 'ddp_spawn', 'ddp_find_unused_parameters_true', 'deepspeed', 'fsdp'],
        help='Distributed training strategy'
    )
    
    # Early stopping arguments
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--datasplit_seed', 
        type=int, 
        default=None, 
        help="Random seed for data splitting (default: None)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--use_zuco1_only',
        action='store_true',
        help='Use only ZuCo1 dataset (filter out ZuCo2)'
    )
    parser.add_argument(
        '--use_channel_weights',
        action='store_true',
        help='Use learnable weights for each EEG channel'
    )
    parser.add_argument(
        '--use_scaled_lr',
        action='store_true',
        help='Scale learning rate by number of GPUs in multi-GPU training'
    )
    parser.add_argument(
        '--use_per_gpu_batch_size',
        action='store_true',
        help='Treat batch_size as per-GPU (global batch size = batch_size × num_gpus)'
    )
    parser.add_argument(
        '--use_class_weights',
        action='store_true',
        help='Use per-class loss weighting based on inverse frequency'
    )

    return parser.parse_args()


def display_label_distributions(datamodule, classification_tasks):
    """Display label distributions for all classification tasks."""
    if not is_main_process():
        return

    print("\n" + "=" * 80)
    print("Multi-Task Label Distributions")
    print("=" * 80)

    datamodule.setup('fit')
    train_dataset = datamodule.train_set

    for task_name, task_labels in classification_tasks.items():
        print(f"\n{task_name.upper()} Task:")
        print("-" * 40)

        label_key = f"{task_name} label" if task_name == "sentiment" else f"{task_name}_label"
        labels = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            label = sample.get(label_key, None)
            if label is not None:
                labels.append(label)

        label_to_id = {label: i for i, label in enumerate(task_labels)}
        label_ids = [label_to_id.get(label, -1) for label in labels]

        label_counts = Counter(label_ids)
        id_to_label = {i: label for i, label in enumerate(task_labels)}

        print(f"Total samples: {len(label_ids)}")
        for label_id in sorted(label_counts.keys()):
            if label_id == -1:
                print(f"  Unknown: {label_counts[label_id]} samples")
                continue
            label_name = id_to_label.get(label_id, f'unknown({label_id})')
            count = label_counts[label_id]
            percentage = (count / len(label_ids)) * 100
            print(f"  {label_name}: {count} samples ({percentage:.2f}%)")

    print("=" * 80)


def compute_class_counts(datamodule, classification_tasks):
    """Compute class counts for all classification tasks.

    Returns:
        Dict mapping task_name -> {class_id: count}
    """
    datamodule.setup('fit')
    train_dataset = datamodule.train_set

    task_counts = {}
    for task_name, task_labels in classification_tasks.items():
        label_key = f"{task_name} label" if task_name == "sentiment" else f"{task_name}_label"
        label_to_id = {label: i for i, label in enumerate(task_labels)}

        counts = {}
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            label = sample.get(label_key, None)
            if label is not None and label in label_to_id:
                class_id = label_to_id[label]
                counts[class_id] = counts.get(class_id, 0) + 1

        task_counts[task_name] = counts

    return task_counts


def display_confusion_matrices(model, classification_tasks):
    """Display confusion matrices for all classification tasks."""
    if not is_main_process():
        return

    for task_name, task_labels in classification_tasks.items():
        cm_attr = f'confusion_matrix_{task_name}'
        if not hasattr(model, cm_attr):
            print(f"\nWarning: No confusion matrix found for {task_name} task.")
            continue

        cm = getattr(model, cm_attr)
        num_classes = len(task_labels)

        print("\n" + "=" * 80)
        print(f"Confusion Matrix - {task_name.upper()} Task")
        print("=" * 80)
        print("\nRows=true labels, Columns=predictions")
        print(f"Labels: {', '.join([f'{i}={label}' for i, label in enumerate(task_labels)])}\n")

        print("           Predicted")
        print("         ", end="")
        for i in range(num_classes):
            print(f"{i:>8}", end="")
        print()

        print("Actual")
        for i, row in enumerate(cm):
            print(f"  {i}   ", end="")
            for val in row:
                print(f"{val:>8}", end="")
            print()

        print("\nPer-class metrics:")
        for i, label_name in enumerate(task_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {label_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

        print("=" * 80)


def main():
    """Main training function."""
    args = parse_args()

    # ============================================================================
    # DDP Configuration and LR Scaling
    # ============================================================================
    is_ddp = args.strategy in ['ddp', 'ddp_spawn', 'ddp_find_unused_parameters_true']
    num_devices = len(args.device) if isinstance(args.device, list) else 1

    # Calculate effective batch size based on use_per_gpu_batch_size flag
    if args.use_per_gpu_batch_size and num_devices > 1:
        effective_batch_size = args.batch_size * num_devices
    else:
        effective_batch_size = args.batch_size

    # Apply square root LR scaling for multi-GPU if enabled
    if args.use_scaled_lr and is_ddp and num_devices > 1:
        base_lr = args.lr
        # sclae == 1
        scaled_lr = base_lr
        args.lr = scaled_lr
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"Multi-GPU Configuration")
            print(f"{'='*80}")
            print(f"Number of GPUs: {num_devices}")
            print(f"Per-GPU batch size: {args.batch_size}")
            print(f"Effective global batch size: {effective_batch_size}")
            print(f"Base LR: {base_lr}")
            print(f"Scaled LR (square root scaling): {scaled_lr}")
            print(f"{'='*80}\n")
    elif is_ddp and num_devices > 1:
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"Multi-GPU Configuration")
            print(f"{'='*80}")
            print(f"Number of GPUs: {num_devices}")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Effective global batch size: {effective_batch_size}")
            print(f"Learning rate: {args.lr} (no scaling)")
            print(f"{'='*80}\n")

    L.seed_everything(args.seed, workers=True)
    
    # Create timestamp for this run - must be consistent across all processes
    # Use environment variable to share timestamp across processes in DDP
    if 'RUN_TIMESTAMP' in os.environ:
        timestamp = os.environ['RUN_TIMESTAMP']
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.environ['RUN_TIMESTAMP'] = timestamp
    
    run_name = f'{args.experiment_name}_{timestamp}'
    
    # Set up directory structure: ./logs/<experiment_name>_<timestamp>/
    # - tensorboard/  (TensorBoard logs)
    # - checkpoints/  (Model checkpoints)
    # - training.log  (Console output log)
    run_dir = os.path.join(args.log_dir, run_name)
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_file = os.path.join(run_dir, 'training.log')
    
    # Only create directories and TeeLogger on main process
    if is_main_process():
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save the bash script for reproducibility
        bash_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'run_script', 'run_parallel.sh')
        if os.path.exists(bash_script_path):
            shutil.copy2(bash_script_path, os.path.join(run_dir, 'run_parallel.sh'))
            print(f"Saved training script to: {os.path.join(run_dir, 'run_parallel.sh')}")

        # Set up file logging using TeeLogger (defined at module level)
        sys.stdout = TeeLogger(log_file, sys.stdout)
        sys.stderr = TeeLogger(log_file.replace('.log', '_error.log'), sys.stderr)

        print("=" * 80)
        print("SEMKEY_PARALLEL Training (Multi-Task: 2 Classification + 2 Regression)")
        print("=" * 80)
        print(f"Run directory: {run_dir}")
        print(f"Data path: {args.data_path}")
        print(f"Use ZuCo1 only: {args.use_zuco1_only}")
        print(f"Use channel weights: {args.use_channel_weights}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Sentiment labels: {args.sentiment_labels}")
        print(f"Topic labels: {args.topic_labels}")
        print(f"MLP hidden dims: {args.mlp_hidden_dims}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr} (max), {args.min_lr} (min)")
        print(f"Warmup epochs: {args.warmup_epochs}")
        print(f"Max epochs: {args.max_epochs}")
        print(f"Freeze encoder: {args.freeze_encoder}")
        print(f"Use Prompt: {not args.do_not_use_prompt}")
        print(f"Loss weights - clip: {args.clip_loss_weight}, lm: {args.lm_loss_weight}, commitment: {args.commitment_loss_weight}")
        print(f"  sentiment: {args.sentiment_loss_weight}, topic: {args.topic_loss_weight}, length: {args.length_loss_weight}, surprisal: {args.surprisal_loss_weight}")
        print(f"Strategy: {args.strategy}")
        print(f"Devices: {args.device}")
        print(f"Dataset split: {args.datasplit_seed if args.datasplit_seed is not None else 'Use default'}")
        print("=" * 80)
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if is_main_process():
        print("\nInitializing data module with weighted sampling...")

    # Build classification tasks dict
    classification_tasks = {
        'sentiment': args.sentiment_labels,
        'topic': args.topic_labels
    }

    # Build regression tasks list
    regression_tasks = ['length', 'surprisal']

    # Disable weighted sampler for DDP due to WeightedRandomSampler incompatibility
    use_weighted_sampler = not is_ddp

    datamodule = GLIMDataModule(
        data_path=args.data_path,
        eval_noise_input=False,
        bsz_train=args.batch_size,
        bsz_val=args.val_batch_size,
        bsz_test=args.val_batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=use_weighted_sampler,
        classification_label_keys=['sentiment label', 'topic_label'],
        regression_label_keys=regression_tasks,
        use_zuco1_only=args.use_zuco1_only,
        use_spectral_whitening=args.spectral_whitening,
        use_robust_normalize=args.robust_normalize,
        seed=args.datasplit_seed
    )

    if is_main_process():
        if use_weighted_sampler:
            print("Weighted sampling enabled to handle class imbalance")
        else:
            print("Weighted sampling disabled (incompatible with DDP)")

    # Display distribution of labels
    # !!! Commented out to reduce RAM consumption during training
    # If you'd like to show this, pls help us fix multiple .setup() issue
    # -> BUG: multiple copies of dataset would be loaded when .setup() is called 
    # display_label_distributions(datamodule, classification_tasks)

    if is_main_process():
        print("\nInitializing SEMKEY_PARALLEL model...")

    if args.checkpoint is not None:
        if is_main_process():
            print(f"Loading model from checkpoint: {args.checkpoint}")
        model = SEMKEY_PARALLEL.load_from_checkpoint(args.checkpoint)
    else:
        if is_main_process():
            print("Creating new SEMKEY_PARALLEL model...")
        model = SEMKEY_PARALLEL(
            input_eeg_len=DEFAULT_INPUT_EEG_LEN,
            hidden_eeg_len=args.hidden_eeg_len,
            input_text_len=DEFAULT_INPUT_TEXT_LEN,
            input_dim=DEFAULT_INPUT_DIM,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            text_model_id=args.text_model,
            model_cache_dir=args.model_cache_dir,
            n_in_blocks=args.n_in_blocks,
            n_out_blocks=args.n_out_blocks,
            num_heads=args.num_heads,
            dropout=args.encoder_dropout,
            use_prompt=(not args.do_not_use_prompt),
            use_channel_weights=args.use_channel_weights,
            batch_size=args.batch_size,

            # Multi-task arguments
            classification_tasks=classification_tasks,
            regression_tasks=regression_tasks,
            mlp_hidden_dims=args.mlp_hidden_dims,
            mlp_dropout=args.mlp_dropout,
            freeze_encoder=args.freeze_encoder,

            # Loss weights
            clip_loss_weight=args.clip_loss_weight,
            lm_loss_weight=args.lm_loss_weight,
            commitment_loss_weight=args.commitment_loss_weight,
            sentiment_loss_weight=args.sentiment_loss_weight,
            topic_loss_weight=args.topic_loss_weight,
            length_loss_weight=args.length_loss_weight,
            surprisal_loss_weight=args.surprisal_loss_weight,
            use_class_weights=args.use_class_weights,

            # Optimizer arguments
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
        )
    
    if is_main_process():
        print(f"Model created with:")
        print(f"  Embedding dim: {model.embed_dim}")
        print(f"  MLP architecture: {model.embed_dim} -> {' -> '.join(map(str, args.mlp_hidden_dims))} -> provided cls label count")
        print(f"  * Please notice that the architecture may vary due to the label provided")
        
        if args.freeze_encoder:
            print(f"  Encoder: FROZEN (only training MLP classifier)")
        else:
            print(f"  Encoder: TRAINABLE")

    # Set class weights if enabled
    if args.use_class_weights:
        if is_main_process():
            print("\nComputing class weights for loss weighting...")
        class_counts = compute_class_counts(datamodule, classification_tasks)
        model.set_class_weights(class_counts['sentiment'], class_counts['topic'])
        if is_main_process():
            print(f"  Sentiment weights: {model.sentiment_class_weights.tolist()}")
            print(f"  Topic weights: {model.topic_class_weights.tolist()}")

    # Compute and set regression normalization statistics
    if is_main_process():
        print("\nComputing regression normalization statistics...")
    datamodule.setup('fit')
    train_dataset = datamodule.train_set
    length_values = [train_dataset.data['length'][i] for i in range(len(train_dataset))]
    surprisal_values = [train_dataset.data['surprisal'][i] for i in range(len(train_dataset))]

    import numpy as np
    length_mean = float(np.mean(length_values))
    length_std = float(np.std(length_values))
    surprisal_mean = float(np.mean(surprisal_values))
    surprisal_std = float(np.std(surprisal_values))

    model.set_regression_stats(length_mean, length_std, surprisal_mean, surprisal_std)
    if is_main_process():
        print(f"  Length: mean={length_mean:.2f}, std={length_std:.2f}")
        print(f"  Surprisal: mean={surprisal_mean:.4f}, std={surprisal_std:.4f}")

    # Set up TensorBoard logger (inside tensorboard/ subdirectory)
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir,
        name='',
        version='',
        default_hp_metric=False
    )
    
    if is_main_process():
        print(f"\nTensorBoard logs: {tensorboard_dir}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(f"Training log: {log_file}")
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-epoch{epoch:02d}-acc_sentiment{val/acc_sentiment_correct:.4f}',
            monitor='val/acc_sentiment_correct',
            mode='max',
            save_top_k=1,
            save_last=True,
            verbose=True,
            enable_version_counter=False,
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-epoch{epoch:02d}-acc_topic{val/acc_topic_correct:.4f}',
            monitor='val/acc_topic_correct',
            mode='max',
            save_top_k=1,
            save_last=True,
            verbose=True,
            enable_version_counter=False,
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-epoch{epoch:02d}-loss_length{val/mae_length:.4f}',
            monitor='val/mae_length',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
            enable_version_counter=False,
            auto_insert_metric_name=False
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-epoch{epoch:02d}-loss_surprisal{val/mae_surprisal:.4f}',
            monitor='val/mae_surprisal',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
            enable_version_counter=False,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=10)
    ]
    
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.001,
                patience=args.patience,
                mode='min',
                verbose=True
            )
        )
        if is_main_process():
            print(f"Early stopping enabled with patience={args.patience}")
    
    if is_main_process():
        print("\nInitializing trainer...")
    
    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'auto'

    # Configure DDP strategy with find_unused_parameters for multi-task model
    if is_ddp:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
        if is_main_process():
            print(f"Using DDPStrategy with find_unused_parameters=True")
    else:
        strategy = args.strategy

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=device,
        strategy=strategy,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        deterministic=True,
        use_distributed_sampler=False,  # Custom samplers already handle distributed training
        sync_batchnorm=True  # Synchronize BatchNorm statistics across GPUs
    )
    
    trainer.logger.log_hyperparams(vars(args))
    
    if is_main_process():
        print("\nStarting training...")
        print("=" * 80)
    
    trainer.fit(model, datamodule)
    
    if is_main_process():
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
        print("=" * 80)
    
    if trainer.checkpoint_callback.best_model_path:
        if is_main_process():
            print("\nRunning test evaluation on best model...")
        trainer.test(model, datamodule,
                    ckpt_path=trainer.checkpoint_callback.best_model_path)

        display_confusion_matrices(model, classification_tasks)


if __name__ == '__main__':
    main()