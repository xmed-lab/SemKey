"""
Training script for semkey stage 2 model.
You don't really need this, you can set freeze-stage1 = true
when training the E2E model.
"""

import os
import sys
import atexit
import argparse
import shutil
import torch
import math
from datetime import datetime
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.stage2_dataset import create_stage2_dataloaders
from model.semkey_stage2 import Stage2ReconstructionModel


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

# I don't know why, but seems to be needed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Stage 2 text reconstruction model'
    )

    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the pandas DataFrame pickle file (e.g., stage2.df from predict_semkey_parallel_and_pack.py). '
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading'
    )

    # Label arguments
    parser.add_argument(
        '--sentiment_labels',
        type=str,
        nargs='+',
        default=['non_neutral', 'neutral'],
        help='Sentiment label names (space-separated)'
    )
    parser.add_argument(
        '--topic_labels',
        type=str,
        nargs='+',
        default=['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment'],
        help='Topic label names (space-separated)'
    )
    parser.add_argument(
        '--train_Zi_drop_prob',
        type=float,
        default=-1.0,
        help='Rate to drop Zi when training (default to -1, if number < 0, will not drop)'
    )

    # Model arguments
    parser.add_argument(
        '--text_model',
        type=str,
        default='google/flan-t5-large',
        help='Pre-trained text model to use'
    )
    parser.add_argument(
        '--freeze_strategy',
        type=str,
        default='lora',
        choices=['lora', 'full_freeze_llm', 'full_trainable_llm'],
        help='Freeze strategy: lora or full_freeze_llm or full_trainable_llm'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=8,
        help='LoRA rank'
    )
    parser.add_argument(
        '--label_embed_init',
        type=str,
        default=None,
        help='Path to pre-trained label embeddings (optional)'
    )
    parser.add_argument(
        '--attention_mask_type',
        type=str,
        default='bidirectional',
        choices=['bidirectional', 'causal'],
        help='Cross-attention mask type: bidirectional (full visibility) or causal (sequential)'
    )
    parser.add_argument(
        '--use_ei',
        action='store_true',
        default=True,
        help='Prepend global EEG feature (ei) to sequence features (Zi)'
    )
    parser.add_argument(
        '--no_use_ei',
        action='store_false',
        dest='use_ei',
        help='Do not use global EEG feature (ei)'
    )
    parser.add_argument(
        '--use_projector',
        action='store_true',
        default=True,
        help='Use trainable projection layer for feature alignment'
    )
    parser.add_argument(
        '--no_projector',
        action='store_false',
        dest='use_projector',
        help='Do not use projection layer'
    )
    # Use prompts 'dataset', 'task', 'subject'
    parser.add_argument(
        '--use_metadata',
        action='store_true',
        help='Include dataset/task/subject metadata in prompts'
    )
    # Prompt type
    parser.add_argument(
        '--prompt_type',
        type=str,
        default='default',
        help='Type of prompt to use for training'
    )

    # Training arguments
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Maximum learning rate'
    )
    parser.add_argument(
        '--proj_lr',
        type=float,
        default=1e-4,
        help='Learning rate for projection layer (only used when --use_projector is set)'
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
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay for AdamW'
    )

    # Hardware arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for training'
    )

    # Output arguments
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Base directory for logs'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='stage2',
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, use_metadata=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device
        label_task1 = batch['label_task1'].to(device)
        label_task2 = batch['label_task2'].to(device)
        ei = batch['ei'].to(device)
        Zi = batch['Zi'].to(device)
        target_text = batch['target_text']
        length = batch['length'].to(device)
        surprisal = batch['surprisal'].to(device)
        prompt_dicts = batch.get('prompt_dicts', None) if use_metadata else None

        # Forward pass
        loss = model(
            label_task1, 
            label_task2, 
            length, 
            surprisal, 
            ei, 
            Zi, 
            target_text,
            prompt_dicts=prompt_dicts
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def validate(model, val_loader, device, use_metadata=False):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            label_task1 = batch['label_task1'].to(device)
            label_task2 = batch['label_task2'].to(device)
            ei = batch['ei'].to(device)
            Zi = batch['Zi'].to(device)
            target_text = batch['target_text']
            length = batch['length'].to(device)
            surprisal = batch['surprisal'].to(device)
            prompt_dicts = batch.get('prompt_dicts', None) if use_metadata else None

            loss = model(
                label_task1, 
                label_task2, 
                length, 
                surprisal, 
                ei, 
                Zi, 
                target_text,
                prompt_dicts=prompt_dicts
            )
            total_loss += loss.item()
            num_batches += 1

    # WARNING!!! Using functions from predict stage2
    from inference import predict_stage2
    predictions , _ , _ , _ = predict_stage2.generate_predictions(model, val_loader, device, use_metadata=use_metadata)
    # Compute text generation metrics (BLEU, ROUGE, WER)
    metrics = predict_stage2.compute_metrics(predictions)
    retrieval_metrics_gen = predict_stage2.compute_retrieval_metrics(predictions, top_k = [1, 5, 10], device = device)

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, metrics, retrieval_metrics_gen


def generate_samples(model, val_loader, device, num_samples=3, use_metadata=False):
    """Generate sample predictions."""
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in val_loader:
            label_task1 = batch['label_task1'].to(device)
            label_task2 = batch['label_task2'].to(device)
            ei = batch['ei'].to(device)
            Zi = batch['Zi'].to(device)
            target_text = batch['target_text']
            length = batch['length'].to(device)
            surprisal = batch['surprisal'].to(device)
            prompt_dicts = batch.get('prompt_dicts', None) if use_metadata else None

            # Generate predictions
            predictions = model.generate(
                label_task1, 
                label_task2, 
                length, 
                surprisal, 
                ei, 
                Zi,
                prompt_dicts=prompt_dicts
            )

            # Collect samples
            for i in range(len(predictions)):
                samples.append({
                    'prediction': predictions[i],
                    'target': target_text[i]
                })
                if len(samples) >= num_samples:
                    return samples

    return samples


def save_checkpoint(epoch, val_loss, train_loss, model, optimizer, args, 
                    checkpoint_dir, checkpoint_tracker, max_checkpoints=3,
                    value_to_track = None, mode = 'min', save_last = True):
    """Save checkpoint with top-k management."""

    # Default -> track val loss
    if value_to_track is None:
        value_to_track = val_loss

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model-epoch{epoch:02d}-loss{val_loss:.4f}-tracking{value_to_track:.4f}.pt"
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args)
    }, checkpoint_path)

    # Track for top-k
    checkpoint_tracker.append((value_to_track, epoch, checkpoint_path))
    checkpoint_tracker.sort(key=lambda x: x[0])

    # Save last checkpoint
    if save_last:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        shutil.copy(checkpoint_path, last_path)

    # Remove old checkpoints beyond top-k
    # (Do not remove anything if max_checkpoints <= 0)
    if max_checkpoints > 0:
        if len(checkpoint_tracker) > max_checkpoints:
            if mode == 'min':
                _, _, old_path = checkpoint_tracker.pop()
            elif mode == 'max':
                _, _, old_path = checkpoint_tracker.pop(0)
            else:
                raise f"[ERROR] Unexpected mode {mode}"
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_path}")

    return checkpoint_path


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create timestamped directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.experiment_name}_{timestamp}'
    run_dir = os.path.join(args.log_dir, run_name)
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_file = os.path.join(run_dir, 'training.log')
    error_log_file = os.path.join(run_dir, 'training_error.log')

    # Create directories
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save the bash script for reproducibility
    bash_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'run_script', 'run_stage2.sh')
    if os.path.exists(bash_script_path):
        shutil.copy2(bash_script_path, os.path.join(run_dir, 'run_stage2.sh'))
        print(f"Saved training script to: {os.path.join(run_dir, 'run_stage2.sh')}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Setup TeeLogger for stdout/stderr
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(error_log_file, sys.stderr)

    # Determine number of classes
    num_sentiment_classes = len(args.sentiment_labels)
    num_topic_classes = len(args.topic_labels)

    # Print configuration
    print("=" * 80)
    print("Stage 2 Training Configuration")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Text model: {args.text_model}")
    print(f"Freeze strategy: {args.freeze_strategy}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Attention mask type: {args.attention_mask_type}")
    print(f"Use global EEG (ei): {args.use_ei}")
    print(f"Zi drop prob ( < 0 no drop ): {args.train_Zi_drop_prob}")
    print(f"Use projection layer: {args.use_projector}")
    if args.use_projector:
        print(f"Projection layer LR: {args.proj_lr}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Using metadata in prompts: {args.use_metadata}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.lr} (max), {args.min_lr} (min)")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Device: {args.device}")
    print(f"Sentiment labels ({num_sentiment_classes}): {args.sentiment_labels}")
    print(f"Topic labels ({num_topic_classes}): {args.topic_labels}")
    print("=" * 80)

    # Create dataloaders
    print("\nCreating dataloaders...")
    # Use real dataset from DataFrame
    train_loader, val_loader, test_loader = create_stage2_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sentiment_labels=args.sentiment_labels,
        topic_labels=args.topic_labels,
        num_workers=args.num_workers,
        train_Zi_drop_prob=(args.train_Zi_drop_prob if args.train_Zi_drop_prob >= 0 else None)
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    print("\nInitializing model...")
    model = Stage2ReconstructionModel(
        model_name=args.text_model,
        freeze_strategy=args.freeze_strategy,
        lora_rank=args.lora_rank,
        attention_mask_type=args.attention_mask_type,
        use_ei=args.use_ei,
        use_projector=args.use_projector,
        label_embed_init=None,
        sentiment_labels=args.sentiment_labels,
        topic_labels=args.topic_labels,
        prompt_type=args.prompt_type,
        device=args.device
    )
    print("Model initialized successfully")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Print detailed trainable parameters for debugging
    print("\nTrainable Parameters (for verification):")
    print("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:60s} {str(tuple(param.shape)):20s}")
    print("-" * 80)

    # Create parameter groups
    optimizer_groups = [
        {"params": [p for n, p in model.named_parameters() 
                    if "projector" not in n and p.requires_grad]}
    ]

    if args.use_projector:
        optimizer_groups.append({
            "params": [p for n, p in model.named_parameters() 
                       if "projector" in n and p.requires_grad],
            "lr": args.proj_lr
        })

    # Create optimizer
    optimizer = AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler with warmup
    total_steps = args.max_epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    if args.warmup_epochs > 0:
        # Linear warmup followed by cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6 / args.lr,
            end_factor=1.0,
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
        print(f"\nLR Scheduler: Linear warmup ({args.warmup_epochs} epochs) + Cosine annealing")
    else:
        # Just cosine annealing without warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr
        )
        print(f"\nLR Scheduler: Cosine annealing (no warmup)")

    # Log hyperparameters to TensorBoard
    writer.add_text('config/args', str(vars(args)))

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_val_loss = float('inf')
    checkpoint_tracker = []
    bleu_ckpt_tracker = []

    for epoch in range(1, args.max_epochs + 1):
        print(f"\nEpoch {epoch}/{args.max_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, scheduler, 
            args.device, epoch, use_metadata=args.use_metadata
        )
        current_lr = optimizer.param_groups[0]['lr']
        if args.use_projector:
            current_lr_proj = optimizer.param_groups[1]['lr']
            print(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_ppl:.2f} | LR: {current_lr:.2e} | Proj_LR: {current_lr_proj:.2e}")
        else:
            print(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_ppl:.2f} | LR: {current_lr:.2e}")

        # Log training metrics to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/perplexity', train_ppl, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)

        # Log projector LR if we have projector
        if args.use_projector:
            writer.add_scalar('train/lr_proj', current_lr_proj, epoch)

        # Validate
        val_loss, val_ppl, metrics, retrieval_metrics_gen = validate(
            model, val_loader, args.device, use_metadata=args.use_metadata
        )
        print(f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
        print(f"Top-1 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top01']:.4f}")
        print(f"Top-5 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top05']:.4f}")
        print(f"Top-10 Accuracy:  {retrieval_metrics_gen['retrieval_acc_top10']:.4f}")

        # Log validation metrics to TensorBoard
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/perplexity', val_ppl, epoch)
        writer.add_scalar('val/retrieval_acc_top01', retrieval_metrics_gen['retrieval_acc_top01'], epoch)
        writer.add_scalar('val/retrieval_acc_top05', retrieval_metrics_gen['retrieval_acc_top05'], epoch)
        writer.add_scalar('val/retrieval_acc_top10', retrieval_metrics_gen['retrieval_acc_top10'], epoch)
        writer.add_scalar('val/bleu1', metrics['mean']['bleu1'], epoch)
        writer.add_scalar('val/bleu2', metrics['mean']['bleu2'], epoch)
        writer.add_scalar('val/bleu3', metrics['mean']['bleu3'], epoch)
        writer.add_scalar('val/bleu4', metrics['mean']['bleu4'], epoch)
        writer.add_scalar('val/rouge1_fmeasure', metrics['mean']['rouge1_fmeasure'], epoch)
        writer.add_scalar('val/rouge1_precision', metrics['mean']['rouge1_precision'], epoch)
        writer.add_scalar('val/rouge1_recall', metrics['mean']['rouge1_recall'], epoch)
        writer.add_scalar('val/wer', metrics['mean']['wer'], epoch)

        # Generate sample predictions
        print("\nSample Predictions:")
        samples = generate_samples(
            model, val_loader, args.device, 
            num_samples=3, use_metadata=args.use_metadata
        )
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Target:     {sample['target']}")
            print(f"  Prediction: {sample['prediction']}")

        # Save checkpoint with top-k management
        # Also save bleu recorded ckpts
        checkpoint_path = save_checkpoint(
            epoch, val_loss, train_loss, model, optimizer, args,
            checkpoint_dir, checkpoint_tracker, max_checkpoints=3
        )
        save_checkpoint(
            epoch, val_loss, train_loss, model, optimizer, args,
            checkpoint_dir, bleu_ckpt_tracker, max_checkpoints=5,
            value_to_track = metrics['mean']['bleu1'],
            mode = 'max',
            save_last = False
        )
        print(f"\nCheckpoint saved: {checkpoint_path}")

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set...")
    print("=" * 80)
    test_loss, test_ppl, metrics, retrieval_metrics_gen = validate(
        model, test_loader, args.device, use_metadata=args.use_metadata
    )
    print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")
    print(f"Top-1 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top01']:.4f}")
    print(f"Top-5 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top05']:.4f}")
    print(f"Top-10 Accuracy:  {retrieval_metrics_gen['retrieval_acc_top10']:.4f}")

    # Log test metrics to TensorBoard
    writer.add_scalar('test/loss', test_loss, args.max_epochs)
    writer.add_scalar('test/perplexity', test_ppl, args.max_epochs)
    writer.add_scalar('test/retrieval_acc_top01', retrieval_metrics_gen['retrieval_acc_top01'], args.max_epochs)
    writer.add_scalar('test/retrieval_acc_top05', retrieval_metrics_gen['retrieval_acc_top05'], args.max_epochs)
    writer.add_scalar('test/retrieval_acc_top10', retrieval_metrics_gen['retrieval_acc_top10'], args.max_epochs)
    writer.add_scalar('test/bleu1', metrics['mean']['bleu1'], args.max_epochs)
    writer.add_scalar('test/bleu2', metrics['mean']['bleu2'], args.max_epochs)
    writer.add_scalar('test/bleu3', metrics['mean']['bleu3'], args.max_epochs)
    writer.add_scalar('test/bleu4', metrics['mean']['bleu4'], args.max_epochs)
    writer.add_scalar('test/rouge1_fmeasure', metrics['mean']['rouge1_fmeasure'], args.max_epochs)
    writer.add_scalar('test/rouge1_precision', metrics['mean']['rouge1_precision'], args.max_epochs)
    writer.add_scalar('test/rouge1_recall', metrics['mean']['rouge1_recall'], args.max_epochs)
    writer.add_scalar('test/wer', metrics['mean']['wer'], args.max_epochs)

    # Log hyperparameters with final metrics
    # Convert args to dict with only scalar/string values (TensorBoard add_hparams doesn't support lists)
    hparams = {k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()}
    writer.add_hparams(
        hparams,
        {
            'hparam/best_val_loss': best_val_loss,
            'hparam/test_loss': test_loss,
            'hparam/test_perplexity': test_ppl
        }
    )

    # Close TensorBoard writer
    writer.close()

    # Test ckpts in bleu_ckpt_tracker
    print("\n" + "=" * 80)
    print("Testing higher BLEU-1 checkpoints...")
    print("=" * 80)
    from inference import predict_stage2
    import pandas as pd
    df = pd.read_pickle(args.data_path)
    test_loader, df_split = predict_stage2.create_dataloader(df, 'test', args.batch_size, args.sentiment_labels, args.topic_labels)
    for (bleu, epoch, path) in bleu_ckpt_tracker:
        print(f"Testing epoch-{epoch} path-{path}")
        # WARNING!!! Using functions from predict stage2
        model = predict_stage2.load_model_from_checkpoint(path, None, None, None, args.device)
        predictions , _ , _ , _ = predict_stage2.generate_predictions(model, test_loader, args.device, use_metadata=args.use_metadata)
        # Compute text generation metrics (BLEU, ROUGE, WER)
        metrics = predict_stage2.compute_metrics(predictions)
        retrieval_metrics_gen = predict_stage2.compute_retrieval_metrics(predictions, top_k = [1, 5, 10], device = args.device)
         # Add retrieval metrics to the metrics dictionary
        metrics["mean"]["retrieval_eeg_gen_top01"] = retrieval_metrics_gen["retrieval_acc_top01"]
        metrics["mean"]["retrieval_eeg_gen_top05"] = retrieval_metrics_gen["retrieval_acc_top05"]
        metrics["mean"]["retrieval_eeg_gen_top10"] = retrieval_metrics_gen["retrieval_acc_top10"]
        # Save the metrics
        output_path = path.replace(".pt", ".csv")
        predict_stage2.save_predictions(predictions, df_split, output_path, metrics)
        # Verbose
        print(f"Top-1 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top01']:.4f}")
        print(f"Top-5 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top05']:.4f}")
        print(f"Top-10 Accuracy:  {retrieval_metrics_gen['retrieval_acc_top10']:.4f}")
        print("-" * 80)


    print("\nTraining complete!")
    print(f"Logs saved to: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")


if __name__ == '__main__':
    main()
