"""
Training script for SEMKEY_E2E model.
"""

import os
import sys
import atexit
import argparse
import shutil
import torch
from datetime import datetime
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.e2e_datamodule import E2EDataModule
from model.semkey_e2e import SEMKEY_Stage2_E2E


class TeeLogger:
    """Logger that writes to both stdout/stderr and a file."""
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
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def close(self):
        if self.log and not self.log.closed:
            self.log.close()

    @classmethod
    def close_all(cls):
        for instance in cls._instances:
            instance.close()


atexit.register(TeeLogger.close_all)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SemKey Stage 1 + Stage2 model')

    # Stage 1 arguments
    parser.add_argument('--stage1_checkpoint', required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--freeze_stage1', action='store_true', help='Freeze Stage 1 encoder')

    # Stage 2 arguments
    parser.add_argument('--text_model', default='google/flan-t5-large', help='Text model')
    parser.add_argument('--freeze_strategy', default='lora', choices=['lora', 'full_freeze_llm', 'full_trainable_llm'])
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--use_ei', action='store_true', default=True, help='Use global EEG feature')
    parser.add_argument('--no_use_ei', action='store_false', dest='use_ei')
    parser.add_argument('--use_projector', action='store_true', default=True, help='Use projector')
    parser.add_argument('--no_use_projector', action='store_false', dest='use_projector')
    parser.add_argument('--prompt_type', default='default', help='Prompt type')

    # Loss configuration
    parser.add_argument('--use_align_loss', action='store_true', help='Enable alignment losses')
    parser.add_argument('--w_align', type=float, default=0.1, help='Alignment loss weight')
    parser.add_argument('--use_aux_loss', action='store_true', default=True, help='Enable auxiliary losses')
    parser.add_argument('--no_use_aux_loss', action='store_false', dest='use_aux_loss')
    parser.add_argument('--w_llm', type=float, default=1.0, help='LLM loss weight')
    parser.add_argument('--w_sentiment', type=float, default=0.25, help='Sentiment loss weight')
    parser.add_argument('--w_topic', type=float, default=0.25, help='Topic loss weight')
    parser.add_argument('--w_length', type=float, default=0.25, help='Length loss weight')
    parser.add_argument('--w_surprisal', type=float, default=0.25, help='Surprisal loss weight')

    # Optimizer configuration
    parser.add_argument('--s1_lr', type=float, default=1e-5, help='Stage 1 learning rate')
    parser.add_argument('--proj_lr', type=float, default=1e-4, help='Projector learning rate')
    parser.add_argument('--llm_lr', type=float, default=1e-4, help='LLM learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

    # Data arguments
    parser.add_argument('--eeg_data_path', required=True, help='Path to EEG dataframe')
    parser.add_argument('--labels_data_path', required=True, help='Path to labels dataframe')
    parser.add_argument('--use_mtv', action='store_true', help='Use MTV for training data augmentation')
    parser.add_argument('--spectral_whitening', action='store_true', default=True, help='Apply spectral whitening')
    parser.add_argument('--no_spectral_whitening', action='store_false', dest='spectral_whitening')
    parser.add_argument('--robust_normalize', action='store_true', default=True, help='Apply robust normalization')
    parser.add_argument('--no_robust_normalize', action='store_false', dest='robust_normalize')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum epochs')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='GPU device(s)')
    parser.add_argument('--log_dir', default='./logs/e2e', help='Log directory')
    parser.add_argument('--experiment_name', default='semkey_stage2_e2e', help='Experiment name')
    parser.add_argument('--model_cache_dir', default=None, help='Model cache directory')

    # Seeding
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    parser.add_argument('--datasplit_seed', type=int, default=None, help="Random seed for data splitting (default: None)")

    return parser.parse_args()


def move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def setup_optimizer(model, args):
    """
    Setup optimizer with 3 parameter groups.
    (stage1 + projection + language model)
    """
    param_groups = []

    # Group 1: Stage 1 encoder (if not frozen)
    if not args.freeze_stage1:
        s1_params = (
            list(model.stage1.p_embedder.parameters()) +
            list(model.stage1.eeg_encoder.parameters()) +
            list(model.stage1.aligner.parameters())
        )
        if args.use_aux_loss:
            s1_params += (
                list(model.stage1.sentiment_classifier.parameters()) +
                list(model.stage1.topic_classifier.parameters()) +
                list(model.stage1.length_regressor.parameters()) +
                list(model.stage1.surprisal_regressor.parameters())
            )
        param_groups.append({'params': s1_params, 'lr': args.s1_lr})

    # Group 2: Stage 2 projector
    if model.stage2.use_projector:
        param_groups.append({
            'params': model.stage2.projector.parameters(),
            'lr': args.proj_lr
        })

    # Group 3: Stage 2 LLM/LoRA
    llm_params = [p for p in model.stage2.model.parameters() if p.requires_grad]
    param_groups.append({'params': llm_params, 'lr': args.llm_lr})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def setup_scheduler(optimizer, args, total_steps):
    warmup_steps = args.warmup_epochs * (total_steps // args.max_epochs)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6 / args.llm_lr,
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    metrics = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)
        loss = outputs['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        for key, value in outputs.items():
            if key not in metrics:
                metrics[key] = 0
            metrics[key] += value.item()

        pbar.set_postfix({'loss': loss.item()})

    num_batches = len(dataloader)
    metrics = {k: v / num_batches for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    metrics = {}

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)

        for key, value in outputs.items():
            if key not in metrics:
                metrics[key] = 0
            metrics[key] += value.item()

    num_batches = len(dataloader)
    metrics = {k: v / num_batches for k, v in metrics.items()}

    return metrics


@torch.no_grad()
def test_and_save_predictions(model, dataloader, device, save_path):
    """
    Generate predictions on test set and save to CSV.
    (Extendable to save other info ...)
    """
    import csv

    model.eval()
    predictions = []
    targets = []

    for batch in tqdm(dataloader, desc="Testing"):
        batch = move_batch_to_device(batch, device)
        generated_texts = model.generate(batch)

        predictions.extend(generated_texts)
        targets.extend(batch['target text'])

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'target'])
        for pred, target in zip(predictions, targets):
            writer.writerow([pred, target])

    print(f"Saved predictions to {save_path}")


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def main():
    args = parse_args()

    # Set-up seed
    if args.seed is not None:
        # Set up seed for all here
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        # Verbose
        print(f"[DEBUG] Set all seeds to {args.seed}")

    # Dataset seed (verbose)
    if args.datasplit_seed is not None:
        print(f"[DEBUG] Using dataset split seed {args.datasplit_seed}, make you you have run with that split seed in semkey parallel")

    # Setup logging
    log_dir = Path(args.log_dir) / f"{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save the bash script for reproducibility
    bash_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'run_script', 'run_e2e.sh')
    if os.path.exists(bash_script_path):
        shutil.copy2(bash_script_path, os.path.join(log_dir, 'run_e2e.sh'))
        print(f"Saved training script to: {os.path.join(log_dir, 'run_e2e.sh')}")

    sys.stdout = TeeLogger(log_dir / 'training.log', sys.stdout)
    sys.stderr = TeeLogger(log_dir / 'training_error.log', sys.stderr)

    writer = SummaryWriter(log_dir / 'tensorboard')

    device = torch.device(f"cuda:{args.device[0]}" if torch.cuda.is_available() else "cpu")

    # Load data
    datamodule = E2EDataModule(
        eeg_data_path=args.eeg_data_path,
        labels_data_path=args.labels_data_path,
        use_mtv=args.use_mtv,
        spectral_whitening=args.spectral_whitening,
        robust_normalize=args.robust_normalize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.datasplit_seed
    )
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    # Initialize model
    stage2_config = {
        'model_name': args.text_model,
        'freeze_strategy': args.freeze_strategy,
        'lora_rank': args.lora_rank,
        'use_ei': args.use_ei,
        'use_projector': args.use_projector,
        'prompt_type': args.prompt_type,
        'device': str(device),
        'cache_dir': args.model_cache_dir
    }

    loss_config = {
        'use_align_loss': args.use_align_loss,
        'use_aux_loss': args.use_aux_loss,
        'w_align': args.w_align,
        'w_llm': args.w_llm,
        'w_sentiment': args.w_sentiment,
        'w_topic': args.w_topic,
        'w_length': args.w_length,
        'w_surprisal': args.w_surprisal
    }

    model = SEMKEY_Stage2_E2E(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_config=stage2_config,
        loss_config=loss_config
    ).to(device)

    optimizer = setup_optimizer(model, args)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = setup_scheduler(optimizer, args, total_steps)

    best_val_loss = float('inf')

    for epoch in range(args.max_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)

        val_metrics = evaluate(model, val_loader, device)

        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)

        checkpoint_dir = log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            checkpoint_dir / 'last.pt'
        )

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / 'best_loss.pt'
            )

        # Generate and save test predictions
        test_and_save_predictions(
            model, test_loader, device,
            checkpoint_dir / f'e2e_epoch{epoch}.csv'
        )

    writer.close()


if __name__ == '__main__':
    main()
