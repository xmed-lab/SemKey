# Evaluation Script for SEMKEY
# - This script computes a comprehensive set of metrics for
#   evaluating text generation quality, including
#   - traditional n-gram metrics, 
#   - semantic similarity,
#   - retrieval accuracy
#   - and richness/diversity measures.

# Import necessary libraries
import pandas as pd
import numpy as np
import scipy
from collections import Counter
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from sentence_transformers import SentenceTransformer
from pycocoevalcap.cider.cider import Cider

# Type hinting
from typing import List, Dict, Tuple

# Some Macros
PATH_TO_VARIANTS = \
    './data/zuco_preprocessed_dataframe/zuco_label_8variants.df'
# Variant keys (MTV) for BAD BAD BLEU calculation
VARIANT_KEYS = \
    ['lexical simplification (v0)', 'lexical simplification (v1)',
    'semantic clarity (v0)', 'semantic clarity (v1)',
    'syntax simplification (v0)', 'syntax simplification (v1)',
    'naive rewritten', 'naive simplified']
# Comment above and uncomment below to disable calculation on MTV
# VARIANT_KEYS = ['input text']
NLTK_DATA = './data/zuco_preprocessed_dataframe/nltk_data'

# Ensure NLTK resources are available
def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'punkt_tab', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            if res == 'stopwords': nltk.data.find('corpora/stopwords')
            elif res in ['wordnet', 'omw-1.4']: nltk.data.find(f'corpora/{res}')
            else: nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Downloading NLTK resource: {res}...")
            nltk.download(res, quiet = True, download_dir = NLTK_DATA)

# Retrieval Metrics
def compute_retrieval_metrics(preds: List[str], targets: List[str],
                              device: str = "cuda",
                              n_ways: List[int] = [2, 4, 10, 24],
                              num_trials: int = 500) \
                            -> Dict[str, float]:
    
    """
    Compute retrieval-based accuracy metrics.
    (we assume that sentences are repeated equally, true for ZuCo)
    Arguments:
        - preds: List of predicted sentences.
        - targets: List of target sentences (must be same length as preds).
        - device: 'cuda' or 'cpu' for embedding computation.
        - n_ways: List of N values for N-way retrieval evaluation.
        - num_trials: Number of random trials to average over for each N-way setting.
    Returns:
        - Dictionary mapping each N-way setting to its average accuracy.
    """

    print("\033[1m>>> Calculating Retrieval Metrics (N-way Classification) >>>\033[0m")
    
    # Initialize SBERT model
    model = SentenceTransformer('all-mpnet-base-v2', device = device)
    
    # Encode all predictions and targets
    print("  - Encoding predictions and targets.")
    pred_emb = model.encode(preds, batch_size = 64, convert_to_tensor = True, show_progress_bar = False, normalize_embeddings = True)
    target_emb = model.encode(targets, batch_size = 64, convert_to_tensor = True, show_progress_bar = False, normalize_embeddings = True)
    # assert we have the same shape
    assert pred_emb.shape == target_emb.shape, "[ERROR] Pred and Target embedding shapes mismatch"
    
    # Get total number of samples
    total_samples = pred_emb.shape[0]

    # Initialize results dictionary
    results = { }

    # For each N in n_ways, perform retrieval evaluation
    for n in n_ways:

        # Check if we have enough samples to perform N-way retrieval
        if total_samples < n:
            print(f"  \033[93m[Warning] Not enough samples ({total_samples}) for {n}-way retrieval. Skipping.\033[0m")
            results[f"{n}-way"] = 0.0
            continue
            
        print(f"  - Computing {n}-way accuracy (averaging over {num_trials} trials).")
        
        # Initialize list to store accuracies for each trial    
        accuracies = []
        
        # Set seed (for reproducibility)
        torch.random.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Perform multiple trials to get a *stable* estimate
        for _ in range(num_trials):

            # Randomly sample N indices from the dataset
            indices = torch.randperm(total_samples)[:n]
            
            # Get the corresponding embeddings for the sampled indices
            batch_pred = pred_emb[indices].clone()
            batch_target = target_emb[indices]

            # Calculate sim matrix and locate the same one
            # NOTE: sentences are repeated equally, with multiple trails the result is proven stable
            sim_matrix = torch.mm(batch_pred, batch_target.transpose(0, 1))
            
            # Get Top-1 prediction
            predictions = sim_matrix.argmax(dim = 1)

            # Create ground truth labels (0 to n-1)
            ground_truth = torch.arange(n, device = device)
            # Get correct count
            correct_count = (predictions == ground_truth).sum().item()
            
            # Calculate accuracy for this trial and store it
            acc = correct_count / n
            accuracies.append(acc)
            
        # Average accuracy over all trials for this N-way setting
        results[f"{n}-way"] = sum(accuracies) / len(accuracies)

    return results

# Traditional Metrics (BLEU, ROUGE, WER)
def compute_traditional_metrics(preds: List[str], targets: List[str]) -> Dict[str, float]:
    
    """
    Compute traditional n-gram based metrics: BLEU (1-4), ROUGE-1, and WER.
    For BLEU, we use multiple variants of the target sentence as references (if available).
    Arguments:
        - preds: List of predicted sentences.
        - targets: List of target sentences (must be same length as preds).
    Returns:
        - Dictionary containing BLEU-1 to BLEU-4, ROUGE-1 F1, and WER scores.
    """

    print("\033[1m>>> Calculating Traditional Metrics (BLEU, ROUGE, WER) >>>\033[0m")
    
    # Load variants dataframe (for BLEU reference construction)
    print(f"  - Loading variants from {PATH_TO_VARIANTS}.")
    try:
        var_df = pd.read_pickle(PATH_TO_VARIANTS)
    except Exception as e:
        print(f"  \033[91m[Error] Failed to load variants file: {e}\033[0m")
        return { }

    # Pre-process data to construct reference lists for BLEU and ROUGE/WER
    print(" - Pre-processing data for metrics calculation.")
    refs_for_bleu: List[List[str]] = [ ]
    refs_for_rouge_wer: List[str] = [ ]
    clean_preds: List[str] = [ ]

    # Iterate through predictions and targets to build reference lists
    for pred, target in tqdm(zip(preds, targets), total = len(preds), desc = "Aligning Variants"):
        
        # Clean predictions and targets (handle None or empty strings)
        p_clean = " " if pred is None or not pred.strip() else pred
        t_clean = " " if target is None or not target.strip() else target

        # Find the corresponding row in the variants dataframe using the input text (target)
        matched_rows = var_df[var_df['input text'] == t_clean]
        if len(matched_rows) == 0:
            raise "[ERROR] Input text not found, check if you supplied raw text"
        else:
            row = matched_rows.iloc[0]
            current_raw = row['raw text']
            current_variants = [row[key] for key in VARIANT_KEYS if key in row and isinstance(row[key], str)]
            if not current_variants: current_variants = [current_raw]
        # Append cleaned prediction and corresponding references for BLEU and ROUGE/WER
        clean_preds.append(p_clean)
        refs_for_bleu.append(current_variants)
        refs_for_rouge_wer.append(current_raw)

    # Calculate BLEU scores for n-grams 1 to 4
    print("  - Calculating BLEU scores (1-4).")
    metrics = { }
    for n in [1, 2, 3, 4]:
        scores = []
        for pred, refs in zip(clean_preds, refs_for_bleu):
            try:
                score = bleu_score([pred], [refs], n_gram = n).item()
            except:
                score = 0.0
            scores.append(score)
        metrics[f'bleu{n}'] = sum(scores) / len(scores) if scores else 0.0

        # --- IGNORE ---
        # NOTE: you can use this FASTER version, btw.
        # try:
        #     score = bleu_score(clean_preds, refs_for_bleu, n_gram = n)
        #     metrics[f'bleu{n}'] = score.item()
        # except: metrics[f'bleu{n}'] = 0.0

    # Calculate ROUGE-1 F1, Precision, Recall
    print("  - Calculating ROUGE-1 scores.")
    try:
        rouge_res = rouge_score(clean_preds, refs_for_rouge_wer, rouge_keys = 'rouge1')
        metrics['rouge1_fmeasure'] = rouge_res['rouge1_fmeasure'].item()
        metrics['rouge1_precision'] = rouge_res['rouge1_precision'].item()
        metrics['rouge1_recall'] = rouge_res['rouge1_recall'].item()
    except: metrics['rouge1_fmeasure'] = 0.0

    # Calculate WER
    print("  - Calculating WER.")
    try:
        wer_val = word_error_rate(clean_preds, refs_for_rouge_wer)
        metrics['wer'] = wer_val.item()
    except: metrics['wer'] = 1.0

    return metrics

# Richness & Diversity Metrics
def compute_richness_metrics(preds: List[str], targets: List[str]) -> Dict[str, float]:

    """
    Compute richness and diversity metrics
    Arguments:
    - preds: List of predicted sentences.
    - targets: List of target sentences (must be same length as preds).
    Returns:
    - Dictionary containing Dist-1, Dist-2, Content Word Recall, and Head Entropy.
    """

    print("\033[1m>>> Calculating Richness & Diversity Metrics >>>\033[0m")

    # Define a set of stop words for content word recall
    stop_words = set(stopwords.words('english'))

    # Helper function to calculate Dist-N (proportion of unique n-grams)
    def _dist_n(sentences: List[str], n: int) -> float:
        all_ngrams: List[Tuple[str]] = []
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            if len(tokens) < n: continue
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0

    # Calculate Content Word Recall
    total_recall = 0; valid_count = 0
    for pred, ref in zip(preds, targets):
        ref_content = [w.lower() for w in word_tokenize(ref) if w.isalnum() and w.lower() not in stop_words]
        pred_content = [w.lower() for w in word_tokenize(pred) if w.isalnum() and w.lower() not in stop_words]
        if not ref_content: continue
        total_recall += sum((Counter(pred_content) & Counter(ref_content)).values()) / len(ref_content)
        valid_count += 1
    
    # Calculate Head Entropy
    prefixes = [tuple(word_tokenize(s.lower())[ : 2]) for s in preds if len(word_tokenize(s)) >= 2]
    entropy = - sum((c / sum(Counter(prefixes).values())) * np.log2(c / sum(Counter(prefixes).values())) for c in Counter(prefixes).values()) if prefixes else 0.0

    return { "dist_1": _dist_n(preds, 1), 
             "dist_2": _dist_n(preds, 2), 
             "content_recall": total_recall / valid_count if valid_count > 0 else 0,
             "head_entropy": entropy }

# Semantic Similarity Metrics (BERTScore, SBERT Cosine Similarity)
def get_bertscore_similarity(preds: List[str], targets: List[str], 
                             device: str = "cuda",
                             model_type: str = 'roberta-large', 
                             batch_size: int = 64) -> float:

    print("\033[1m>>> Calculating BERTScore Similarity >>>\033[0m")

    # Load BERTScore model
    print(f"  - Loading BERTScore model: {model_type} on {device}.")
    try:
        _ , _ , F1 = score(cands = preds, refs = targets,
                           model_type = model_type,
                           lang = "en",
                           verbose = False,
                           batch_size = batch_size,
                           device = device)
        return F1.mean().item()
    except Exception as e:
        print(f"  \033[91mBERTScore error: {e}\033[0m")
        return 0.0

# SBERT Cosine Similarity
def get_sentence_transformer_similarity(preds: List[str], targets: List[str],
                                        device: str = "cuda",
                                        model_name: str = 'all-mpnet-base-v2',
                                        batch_size: int = 64) -> float:

    print("\033[1m>>> Calculating SBERT Cosine Similarity >>>\033[0m")

    # Load SBERT model
    print(f"  - Loading SBERT model: {model_name} on {device}.")
    model = SentenceTransformer(model_name, device = device)

    # Encode predictions and targets to get embeddings
    print("  - Encoding predictions and targets for Cosine Sim.")
    embeddings_preds = model.encode(preds, batch_size = batch_size, convert_to_tensor = True, show_progress_bar = False, device = device)
    embeddings_targets = model.encode(targets, batch_size = batch_size, convert_to_tensor = True, show_progress_bar = False, device = device)
    cosine_scores = F.cosine_similarity(embeddings_preds, embeddings_targets, dim = 1)
    
    return cosine_scores.mean().item()

# Compute METEOR
def compute_meteor(preds: List[str], targets: List[str]) -> float:
    
    scores = []
    for pred, target in zip(preds, targets):
        try:
            pred_tokens = word_tokenize(pred.lower())
            target_tokens = word_tokenize(target.lower())
            score = meteor_score([target_tokens], pred_tokens)
            scores.append(score)
        except:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

# Compute CIDEr
def compute_cider(preds: List[str], targets: List[str]) -> float:
    try:
        gts = {i: [targets[i]] for i in range(len(targets))}
        res = {i: [preds[i]] for i in range(len(preds))}
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return score
    except Exception as e:
        print(f"  \033[93m[Warning] CIDEr computation failed: {e}\033[0m")
        return 0.0

# Compute Self-BLEU (Diversity) - THIS IS SLOW, USE WITH CAUTION
def compute_self_bleu(preds: List[str], n_gram: int = 4) -> float:
    
    # WARNING: This is O(N^2) and can be very slow for large datasets.
    # We will sample a subset if too large.
    if len(preds) > 1000:
        preds = random.sample(preds, 1000)

    # Tokenize predictions for BLEU calculation
    tokenized_preds = [ word_tokenize(p.lower()) for p in preds ]

    # Calculate
    scores: List[float] = [ ]
    for i in tqdm(range(len(tokenized_preds)), desc = "Computing Self-BLEU"):
        refs = [tokenized_preds[j] for j in range(len(tokenized_preds)) if j != i]
        if not refs:
            continue
        try:
            score = bleu_score([" ".join(tokenized_preds[i])], [[" ".join(r) for r in refs]], n_gram = n_gram).item()
            scores.append(score)
        except:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

# Compute Length MAE (Mean Absolute Error of sentence lengths in words)
def compute_length_mae(preds: List[str], targets: List[str]) -> float:
    length_diffs = []
    for pred, target in zip(preds, targets):
        # Fix: Tokenize to count words instead of characters
        # Handling None or empty strings safely
        p_tokens = word_tokenize(pred) if (pred and isinstance(pred, str)) else []
        t_tokens = word_tokenize(target) if (target and isinstance(target, str)) else []
        
        length_diffs.append(abs(len(p_tokens) - len(t_tokens)))
        
    return sum(length_diffs) / len(length_diffs) if length_diffs else 0.0

# Compute Fréchet Distance between prediction and target embeddings
def calculate_frechet_distance(act1: np.ndarray, act2: np.ndarray) -> float:
    
    # Calculate mean and covariance for both sets of activations
    mu1, sigma1 = np.mean(act1, axis = 0), np.cov(act1, rowvar = False)
    mu2, sigma2 = np.mean(act2, axis = 0), np.cov(act2, rowvar = False)
    
    # Calculate squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Fix 1: Add epsilon for numerical stability
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    
    # Calculate sqrt of product
    covmean: np.ndarray = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    # Fix 2: Handle imaginary numbers strictly
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
            m = np.max(np.abs(covmean.imag))
            # print(f"  \033[93m[Warning] Imaginary component {m} in FD\033[0m")
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    # Fix 3: Ensure non-negative result
    return max(0.0, fid)

# Compute Fréchet Distance between prediction and target embeddings using SBERT
def compute_frechet_distance(preds: List[str], targets: List[str],
                             device: str = "cuda",
                             model_name: str = 'all-mpnet-base-v2',
                             batch_size: int = 64) -> float:
   
    print(f"    - Computing Fréchet Distance using {model_name} on {device}.")
    model = SentenceTransformer(model_name, device = device)

    # Encode predictions and targets
    pred_embeddings = model.encode(preds, batch_size = batch_size, convert_to_tensor = False, show_progress_bar = False)
    target_embeddings = model.encode(targets, batch_size = batch_size, convert_to_tensor = False, show_progress_bar = False)

    # Calculate FD
    fd = calculate_frechet_distance(pred_embeddings, target_embeddings)

    return fd

# Aggregate all metrics into a single function for easy evaluation
def get_all_metrics(preds: List[str], targets: List[str],
                    device: str = "cuda")\
                    -> Dict[str, float]:
    
    print("\n\033[1;32m--- Starting Evaluation Pipeline ---\033[0m")
    print("\n[Metric 1/5] Calculating Traditional Metrics.")
    traditional_metrics = compute_traditional_metrics(preds, targets)
    print("\n[Metric 2/5] Calculating BERTScore.")
    bertscore_f1 = get_bertscore_similarity(preds, targets, device = device)
    print("\n[Metric 3/5] Calculating SBERT Cosine Similarity.")
    sentencebert_cosine = get_sentence_transformer_similarity(preds, targets, device = device)
    print("\n[Metric 4/5] Calculating Richness & Diversity.")
    richness_metrics = compute_richness_metrics(preds, targets)

    # Retrieval Metrics
    print("\n[Metric 5/5] Calculating Retrieval Metrics.")
    retrieval_metrics = compute_retrieval_metrics(preds, targets, n_ways=[2, 4, 10, 24], num_trials=10000)

    # Additional NLG Metrics
    print("\n\033[1;32mComputing additional NLG metrics.\033[0m")
    print("  - METEOR.")
    meteor_score = compute_meteor(preds, targets)
    print("  - CIDEr.")
    cider_score = compute_cider(preds, targets)
    # NOTE: Self-BLEU is very slow,
    #       we can skip it or sample a subset for calculation
    # print("  - Self-BLEU...")
    # self_bleu_score = compute_self_bleu(preds)
    print("  - Length MAE.")
    length_mae = compute_length_mae(preds, targets)
    print("  - Fréchet Distance.")
    frechet_distance = compute_frechet_distance(preds, targets, device = device)

    return {
        'bertscore_f1': bertscore_f1,
        'sentencebert_cosine': sentencebert_cosine,
        **traditional_metrics,
        **richness_metrics,
        **retrieval_metrics,
        'meteor': meteor_score,
        'cider': cider_score,
        # NOTE: uncomment if you want to include Self-BLEU (diversity),
        # but be aware of the runtime
        # 'self_bleu': self_bleu_score,
        'length_mae': length_mae,
        'frechet_distance': frechet_distance
    }

# Utility function to print metrics in a nice format
def print_metrics_report(name, metrics):
    print(f"\n{'='*25} Results for: {name} {'='*25}")
    print(f"  [Traditional N-gram Metrics]")
    print(f"  BLEU-1             : {metrics['bleu1']:.4f}")
    print(f"  BLEU-2             : {metrics['bleu2']:.4f}")
    print(f"  BLEU-3             : {metrics['bleu3']:.4f}")
    print(f"  BLEU-4             : {metrics['bleu4']:.4f}")
    print(f"  ROUGE-1 (F)        : {metrics['rouge1_fmeasure']:.4f}")
    print(f"  WER                : {metrics['wer']:.4f}")

    print(f"\n  [Semantic Accuracy]")
    print(f"  BERTScore F1       : {metrics['bertscore_f1']:.4f}")
    print(f"  SBERT Cosine Sim   : {metrics['sentencebert_cosine']:.4f}")
    
    print(f"\n  [Retrieval Accuracy (w/ Forced Top-1)]")
    print(f"  2-Way Accuracy     : {metrics.get('2-way', 0):.4f}")
    print(f"  4-Way Accuracy     : {metrics.get('4-way', 0):.4f}")
    print(f"  10-Way Accuracy    : {metrics.get('10-way', 0):.4f}")
    print(f"  24-Way Accuracy    : {metrics.get('24-way', 0):.4f}")

    print(f"\n  [Richness & Diversity]")
    print(f"  Dist-1 (Unigrams)  : {metrics['dist_1']:.4f}")
    print(f"  Dist-2 (Bigrams)   : {metrics['dist_2']:.4f}")
    print(f"  Content Word Recall: {metrics['content_recall']:.4f}")
    print(f"  Head Entropy       : {metrics['head_entropy']:.4f}")

    print(f"\n  [Additional NLG Metrics]")
    print(f"  METEOR             : {metrics.get('meteor', 0):.4f}")
    print(f"  CIDEr              : {metrics.get('cider', 0):.4f}")
    print(f"  Self-BLEU (Div)    : {metrics.get('self_bleu', 0):.4f}")
    print(f"  Length MAE         : {metrics.get('length_mae', 0):.4f}")
    print(f"  Fréchet Distance   : {metrics.get('frechet_distance', 0):.4f}")
    print("="*60)

# Main function
def main() -> None:

    # import argparse to collect device / path to variants / variant keys ...
    # and most importantly, path to the CSV file (to be evaluated)
    import argparse
    parser = argparse.ArgumentParser(description = "Evaluation Script for SEMKEY")
    parser.add_argument("--csv-path", required = True, help = "Path to the CSV file to be evaluated")
    parser.add_argument("--device", default = "cuda", help = "Device for embedding computation (default: cuda)")
    parser.add_argument("--variants-path", default = None, help = "Path to the variants dataframe")
    parser.add_argument("--variant-keys", default = None, nargs = "+", help = "List of variant keys to use for BLEU reference construction")
    parser.add_argument("--nltk-data-path", default = None, help = "Path to NLTK data directory")
    args = parser.parse_args()

    # Reassign global variables if specified
    global PATH_TO_VARIANTS, VARIANT_KEYS
    if args.variants_path:
        PATH_TO_VARIANTS = args.variants_path
    if args.variant_keys:
        VARIANT_KEYS = args.variant_keys
    if args.nltk_data_path:
        NLTK_DATA = args.nltk_data_path
        nltk.data.path.append(args.nltk_data_path)

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    # Read the CSV file
    try:
        df: pd.DataFrame = pd.read_csv(args.csv_path)
        df['prediction'] = df['prediction'].fillna("").astype(str)
        df['target'] = df['target'].fillna("").astype(str)
    except Exception as e:
        print(f"\033[91m[Error] Failed to read CSV file: {e}\033[0m")
        return

    # Define process function (We keep text after 'Target: ')
    def process_text(text: str) -> str:
        if "Target:" in text:
            return text.split("Target:")[-1].strip()
        return text.strip()
    df['prediction'] = df['prediction'].apply(process_text)

    # Calculate all metrics
    try:
        metrics: Dict[str, float] = get_all_metrics(df['prediction'].tolist(), df['target'].tolist(), device = args.device)
        print_metrics_report("Evaluated Model", metrics)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n\033[91m[Error] Evaluation failed: {e}\033[0m")

    # Save result as json file next to the CSV
    import json, os
    result_path = os.path.splitext(args.csv_path)[0] + "_evaluation_results.json"
    try:
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent = 4)
        print(f"\n\033[1;32mEvaluation results saved to: {result_path}\033[0m")
    except Exception as e:
        print(f"\033[91m[Error] Failed to save evaluation results: {e}\033[0m")

    # ALL DONE!

if __name__ == "__main__":
    main()