#!/usr/bin/env python
"""
Utility code for additional EEG signal processing
-> Whitening
-> Normalization

"""

import numpy as np

def robust_normalize_padded(eeg_data, axis=-1, clip_value=10.0, epsilon=1e-6):
    """
    Performs robust normalization to Zero-Padded / standarded-length EEG signals.
    1. Mask 0 values
    2. Calc var/std on none-zero entries
    3. Z-score normalization
    4. Fill masked 0 back
    
    Args:
        eeg_data (np.ndarray): Input EEG signal, shape: (Channels, 1280)
        axis (int): time axis (seq_len), normally -1
        clip_value (float): signal clip value (-clip, +clip)
        epsilon (float): avoid 0 division value
    
    Returns:
        np.ndarray: Same shape as eeg_data, shape: (Channels, 1280)
    
    WARNING: The returned array might be in float64, check or convert if
             this causes problems.
    """
    # Create masked array
    masked_data = np.ma.masked_equal(eeg_data, 0)
    # [debug] -> print numbers of (tailing) 0
    # print("Number of masked (zero) elements:", np.sum(masked_data.mask))

    # Calc avg. and std.
    mean = np.mean(masked_data, axis=axis, keepdims=True)
    std = np.std(masked_data, axis=axis, keepdims=True)
    
    # Z-score Normalization
    # Note: previously masked data is still MASKED here:
    normalized_masked = (masked_data - mean) / (std + epsilon)
    
    # Clipping
    if clip_value is not None:
        normalized_masked = np.ma.clip(normalized_masked, -clip_value, clip_value)
    
    # Fill previously masked as 0 (maintain same shape as input)
    final_data = normalized_masked.filled(0)
    
    return final_data


def spectral_whitening(eeg_data, alpha=0.95):
    """
    Performs whitening on EEG data,
    counters 1/f noise, signifies high freq. (Gamma)
    
    formula: y[t] = x[t] - alpha * x[t-1]
    
    Args:
        eeg_data (np.ndarray): Input EEG signal
                               shape: (n_channels, n_timepoints) 
                               OR (n_batch, n_channels, n_timepoints)
        alpha (float): Weight, usually 0.95 OR 0.97。
                       note: alpha=0 no change; alpha=1 sim. to First Difference
    
    Returns:
        np.ndarray: Whitened EEG signal (same shape as eeg_data)
    """
    # Ensure floating point type
    eeg_data = eeg_data.astype(np.float32)
    
    # Init. ret. array
    whitened_data = np.zeros_like(eeg_data)
    
    # Handle t = 0
    whitened_data[..., 0] = eeg_data[..., 0]
    
    # Handle t > 0: x[t] - alpha * x[t-1]
    whitened_data[..., 1:] = eeg_data[..., 1:] - alpha * eeg_data[..., :-1]
    
    return whitened_data