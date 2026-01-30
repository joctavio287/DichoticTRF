from typing import List, Union, Sequence, Optional, Dict, Any, Iterable
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path
from scipy import signal
import numpy as np
import logging
import types
import torch
import json
import copy
import mne

import importlib.machinery
import unicodedata
import re

# =================
# GENERAL UTILITIES
def any_on_gpu(
    attributes: list,
) -> bool:
    """
    Check if any of the given attributes require GPU processing.
    Parameters
    ----------
    attributes : list
        List of attribute names.
    Returns
    -------
        bool
            True if any attribute requires GPU, False otherwise.
    """
    gpu_attributes = {
        # 'Phonemes', 'PhonemesDiscrete', 'DNNS'
        'DNNs'
    }
    return any(attr in gpu_attributes for attr in attributes)

def dump_dict_to_json(
    filepath: str,
    data_dict: Union[dict, types.ModuleType],
    create_dirs: bool = True,
    module: bool = False
) -> None:
    """
    Dumps a dictionary to a JSON file. 
    If the output directory does not exist, it is created.
    """
    if module:
        data_dict_ = {name: getattr(data_dict, name) for name in dir(data_dict)}
    else:
        data_dict_ = data_dict
    def _to_jsonable(obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if obj is Ellipsis:
                return "..."
            if obj is NotImplemented:
                return "NotImplemented"
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, importlib.machinery.ModuleSpec):
                return str(obj)
            if isinstance(obj, importlib.machinery.SourceFileLoader):
                return str(obj)
            if isinstance(obj, types.ModuleType):
                return obj.__name__
            # Modules / functions / callables
            if callable(obj) or hasattr(obj, "__name__"):
                return getattr(obj, "__name__", str(obj))
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_to_jsonable(v) for v in obj]
        except Exception:
            return str(obj)
        return obj
    
    output_path = Path(filepath)
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = _to_jsonable(data_dict_)
    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=4)

def load_json_to_dict(
    filepath: str
) -> dict:
    """
    Loads a JSON file into a dictionary.

    Parameters
    ----------
    filepath : str
        The path to the input JSON file.
    
    Returns
    -------
        dict
            The loaded dictionary.
    """
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    return data_dict

def create_default_dict(
    levels: Iterable[Iterable[Any]],
    default_value: Any
) -> Dict:
    """
    Create a nested dictionary from any iterable of iterables (levels).
    The last level is assigned default_value.

    Example:
        levels = (subjects, bands, attributes)
        result[subject][band][attribute] = default_value
    """
    levels = list(levels)
    if not levels:
        return {}

    def _build(idx: int) -> Dict:
        if idx == len(levels) - 1:
            return {key: default_value for key in levels[idx]}
        return {key: _build(idx + 1) for key in levels[idx]}

    return _build(0)

def fill_missing_nested(
    data_name: str,
    data: Dict,
    levels: Iterable[Iterable[Any]],
    default_value: Any,
    log_stage: callable = None,
    logger: logging.Logger = None,
    log_level: str = "info",
    level_names: Iterable[str] = None
) -> Dict:
    """
    Fill missing nested keys in `data` given an iterable of levels.
    The last level gets `default_value`.

    Example:
        levels = (subjects, bands, attributes)
        result[subject][band][attribute] = default_value
    """
    levels = list(levels)
    if level_names is None:
        level_names = [f"level_{i}" for i in range(len(levels))]
    level_names = list(level_names)

    def _fill(d: Dict, idx: int, path: Dict[str, Any]):
        for key in levels[idx]:
            if key not in d:
                if log_stage and logger:
                    label = level_names[idx] if idx < len(level_names) else f"level_{idx}"
                    log_stage(f"{data_name} missing {label} '{key}', adding default structure value --> {default_value}.", logger=logger, level=log_level)
                d[key] = {} if idx < len(levels) - 1 else default_value
            if idx < len(levels) - 1:
                _fill(d[key], idx + 1, {**path, level_names[idx]: key})

    _fill(data, 0, {})
    return data

def clustering_by_correlation(
    data:np.ndarray, 
    axis:int=0
    )->np.ndarray:
    """
    Cluster by correlation the data

    Parameters
    ----------
    data : np.ndarray
        Must be 2D (array-like) where rows are variables and columns are observations
    axis : int, default 0
        Axis to cluster

    Returns
    -------
    np.ndarray
        Ordered indices (missing indices are zero rows)    
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D array-like.")
    data = data.T  if axis == 1 else data

    # Identify zero rows
    null_indexes = np.where(~data.any(axis=1))[0]
    data = data[[i for i in np.arange(data.shape[0]) if i not in null_indexes]]

    # Compute the correlation matrix 
    correlation_matrix = np.corrcoef(data, rowvar=True)

    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - correlation_matrix

    # Ensure the diagonal of the distance matrix is zero and that the matrix is symmetric
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Use squareform to convert the distance matrix to a condensed form
    condensed_distance_matrix = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='single')

    # Get the order of the variables
    ordered_indices = leaves_list(linkage_matrix)
    
    return ordered_indices

def get_gfp_peaks(evoked, min_dist_ms=25, rel_height=0.1):
    """
    Extracts peak latencies and GFP magnitudes from an MNE Evoked object.

    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data container.
    min_dist_ms : float
        Minimum temporal distance between peaks in milliseconds to avoid duplicates.
    rel_height : float
        Relative threshold (0.0 to 1.0) to reject peaks lower than a percentage of the global maximum.

    Returns
    -------
    tuple
        (times, magnitudes) - A tuple containing two numpy arrays:
        - times: Latencies of the peaks in seconds.
        - magnitudes: GFP values at those latencies.
    """
    # Isolate channels and compute Global Field Power (GFP)
    inst = evoked.copy().pick('eeg')
    gfp = np.std(inst.data, axis=0)

    # Convert ms constraint to samples and calculate height threshold
    sfreq = inst.info['sfreq']
    dist_samples = int((min_dist_ms / 1000) * sfreq)
    height_thresh = np.max(gfp) * rel_height

    # Find peaks
    peaks_idx, _ = signal.find_peaks(gfp, distance=dist_samples, height=height_thresh)

    return inst.times[peaks_idx], gfp[peaks_idx]

def get_median_regularization(
    arr: Union[List, np.ndarray]
)-> float:
    """
    Compute the median of an array in log scale.
    Parameters
    ----------
    arr : Union[List, np.ndarray]
        Input array of values.
    Returns
    -------
    float
        The median value in log scale.
    """
    return 10**(np.median(np.log10(np.array(arr))))

def normalize_text(
    text: str
) -> str:
    """
    Normalize text by converting to lowercase, removing accents, punctuation, and extra whitespace.

    Parameters
    ----------
    text : str
        The input text to be normalized.

            Returns
    -------
    str
        The normalized text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove punctuation (keep only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================================
# FILTERING AND RESAMPLING SIGNALS
def get_antialiasing_filter(
    original_sr: int, 
    target_sr: int, 
    cutoff_ratio: float=0.9, 
    gstop_db: float=53
)->np.ndarray:
    """
    Calculate FIR filter coefficients for anti-aliasing before downsampling.

    Parameters
    ----------
    original_sr: int
        Original sampling rate in Hz (e.g., 16000)
    target_sr: int
        Target sampling rate in Hz (e.g., 128)
    cutoff_ratio: float
        What percentage of the target Nyquist frequency to preserve.
            0.90 is safer for TRF than 0.99 (less ringing).
    gstop_db: float
        The stopband attenuation in dB.

    Returns
    -------
    np.ndarray
        The FIR filter coefficients.
    """
    nyquist_target = target_sr / 2.0
    f_pass = nyquist_target * cutoff_ratio
    f_stop = nyquist_target
    transition_width = f_stop - f_pass
    
    # Kaiser window is specifically defined for controlling ripple and transition width
    # Desire attenuation (ripple)
    # Upper bound for the deviation (in dB) of the magnitude of the filter's frequency response from that of the desired filter (not including frequencies in any transition intervals). That is, if w is the frequency expressed as a fraction of the Nyquist frequency, A(w) is the actual frequency response of the filter and D(w) is the desired frequency response, the design requirement is that:
    #         abs(A(w) - D(w))) < 10**(-ripple/20)
    # for 0 <= w <= 1 and w not in a transition interval.
    numtaps, beta = signal.kaiserord(
        ripple=gstop_db, # 
        width=transition_width / (0.5 * original_sr)
    )
    if numtaps % 2 == 0: numtaps += 1
    taps = signal.firwin(
        numtaps=numtaps, 
        cutoff=f_pass, 
        window=('kaiser', beta), 
        fs=original_sr
    )
    return taps

def custom_resample(
    array:np.ndarray, 
    original_sr:int, 
    target_sr:int,
    padtype:str='line',
    axis:int=0
) -> np.ndarray:
    """
    Resample an array from original_sr to target_sr using polyphase filtering.
    
    Parameters
    ----------
    array : np.ndarray
        The input array to be resampled.
    original_sr : int
        The original sampling rate of the array.
    target_sr : int
        The target sampling rate for the resampled array.
    padtype : str, optional
        The type of padding to use. Default is 'line'.
    axis : int, optional
        The axis along which to resample. Default is 0.
    
    Returns
    -------
    np.ndarray
        The resampled array.
    """
    # Calculate upsampling and downsampling factors by finding greatest common divisor
    gcd = np.gcd(int(original_sr), int(target_sr))
    up = int(target_sr // gcd)
    down = int(original_sr // gcd)
    
    if up == 1:
        # Design anti-aliasing filter only when downsampling using firwin
        window_param = taps = get_antialiasing_filter(
            original_sr=original_sr, 
            target_sr=target_sr,
            cutoff_ratio=0.9,
            gstop_db=53
        )
    else:
        window_param = ('kaiser', 5.0) 

    return signal.resample_poly(
        x=array, 
        up=up, 
        down=down, 
        axis=axis, 
        window=window_param, 
        padtype=padtype
    )

def fir_filter(
    array: np.ndarray, 
    sfreq: float, 
    l_freq: Union[float, None] = None, 
    h_freq: Union[float, None] = None,
    axis: int = 0,
    call_type: str = "forward_compensated_cut",
    store_cache: Union[Path, str, None] = None,
    transition_ratio: float = 0.25,
    min_transition_bandwidth: float = 0.5,
    use_fourier: bool = True,
    pass_zero: Union[bool, str] = "bandpass",
    return_delay: bool = True
) -> np.ndarray:
    """
    Apply a FIR filter using the "Two-Stage" (Cascade) logic, standard in EEGLAB.
    
    If both l_freq and h_freq are provided, it operates as a Bandpass filter by designing:
    1. A High-Pass filter (sharp transition, long kernel).
    2. A Low-Pass filter (soft transition, short kernel).
    3. Convolving them to create a single kernel equivalent to applying them sequentially.
    
    If only one frequency is provided, it applies the corresponding single filter (High-Pass for l_freq, Low-Pass for h_freq).

    Parameters
    ----------
    array : np.ndarray
        The input data to be filtered.
    sfreq : float
        The sampling frequency.
    l_freq : float, optional
        High-pass cutoff frequency (e.g., 1 Hz). If None, no high-pass filtering is applied.
    h_freq : float, optional
        Low-pass cutoff frequency (e.g., 40 Hz). If None, no low-pass filtering is applied.
    axis : int, optional
        Axis to filter.
    call_type : str, optional
        Filtering method to use. Options:
        - "both": zero-phase filtering using filtfilt.
        - "forward": forward filtering using lfilter (introduces phase delay).
        - "forward_compensated_cut": forward filtering with delay compensation by cutting final samples.
        - "forward_compensated_reflected": forward filtering with reflected padding to avoid edge artifacts.
    store_cache : Union[Path, str, None], optional
        Path to store filter taps.
    transition_ratio : float, optional
        Ratio of transition bandwidth to cutoff.
    min_transition_bandwidth : float, optional
        Minimum transition bandwidth in Hz.
    use_fourier : bool, optional
        Use FFT convolution for reflected mode. Default is True.
    pass_zero : Union[bool, str], optional
        Type of filter to apply. Default is "bandpass".
        pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}
        Toggles the zero frequency bin (or DC gain) to be in the passband (True) or in the stopband (False).
        'bandstop', 'lowpass' are synonyms for True and 'bandpass', 'highpass' are synonyms for False.
        'lowpass', 'highpass' additionally require cutoff to be a scalar value or a length-one array.
    return_delay : bool, optional
        Whether to return the number of samples the signal is delayed due to phase shift. Default is True.

    Returns
    -------
    if return_delay:
        tuple[int, np.ndarray]
            Number of samples the signal is delayed due to phase shift and the filtered array (if forward_compensated_reflected, then number of samples cut is returned).
    else:
        np.ndarray
            The filtered array.
    """
    # Ensure array is float64 for precision
    if array.dtype != np.float64:
        array = array.astype(np.float64)
    
    # Demean to avoid edge artifacts
    dc_offset = array.mean(axis=axis, keepdims=True)
    array = array - dc_offset
    
    number_of_dims = array.ndim

    # Validate inputs
    if l_freq is None and h_freq is None:
        raise ValueError("At least one of l_freq or h_freq must be provided.")
    if l_freq is not None:
        assert l_freq >= 0.1, "l_freq must be >= 0.1 Hz."
    is_bandpass = (l_freq is not None) and (h_freq is not None)

    # High-Pass Transition (if l_freq exists)
    if l_freq is not None:
        if l_freq <= 2.0:
            l_trans = min_transition_bandwidth 
        else:
            l_trans = min(
                max(l_freq * transition_ratio, 2.0), 
                l_freq
            )
        # Ballanger/Kaiser formula for transition width
        numtaps_hp = int(3.3 / (l_trans / sfreq))
        if numtaps_hp % 2 == 0: numtaps_hp += 1

    # Low-Pass Transition (if h_freq exists)
    if h_freq is not None:
        nyquist = sfreq / 2.0
        h_trans = min(
            max(h_freq * transition_ratio, 2.0), 
            nyquist - h_freq
        )
        # Ballanger/Kaiser formula for transition width
        numtaps_lp = int(3.3 / (h_trans / sfreq))
        if numtaps_lp % 2 == 0: numtaps_lp += 1

    if store_cache:
        store_cache = Path(store_cache)
        if store_cache.exists():
            taps = np.load(store_cache)
        else:
            if is_bandpass:
                taps_hp = signal.firwin(
                    numtaps=numtaps_hp, 
                    cutoff=l_freq, 
                    pass_zero='highpass',  # Blocks DC
                    window='hamming', 
                    fs=sfreq
                )
                
                taps_lp = signal.firwin(
                    numtaps=numtaps_lp, 
                    cutoff=h_freq, 
                    pass_zero='lowpass', # Blocks Nyquist
                    window='hamming',  
                    fs=sfreq
                )
                # Convolve to create Bandpass
                taps = signal.convolve(taps_hp, taps_lp)
            elif l_freq is not None:
                taps = signal.firwin(
                    numtaps=numtaps_hp, 
                    cutoff=l_freq, 
                    pass_zero=pass_zero, 
                    window='hamming', 
                    fs=sfreq
                )
            elif h_freq is not None:
                # Single Low-Pass
                if pass_zero == 'bandpass':
                    pz = 'lowpass'
                    print("Warning: Changing pass_zero from 'bandpass' to 'lowpass' for single low-pass filter.")
                else:
                    pz = pass_zero

                taps = signal.firwin(
                    numtaps=numtaps_lp, 
                    cutoff=h_freq, 
                    pass_zero=pz, 
                    window='hamming', 
                    fs=sfreq
                )
            
            np.save(store_cache, taps)
    else:
        if is_bandpass:
            taps_hp = signal.firwin(numtaps=numtaps_hp, cutoff=l_freq, pass_zero='highpass', window='hamming', fs=sfreq)
            taps_lp = signal.firwin(numtaps=numtaps_lp, cutoff=h_freq, pass_zero='lowpass', window='hamming', fs=sfreq)
            taps = signal.convolve(taps_hp, taps_lp)
        elif l_freq is not None:
            taps = signal.firwin(numtaps=numtaps_hp, cutoff=l_freq, pass_zero=pass_zero, window='hamming', fs=sfreq)
        elif h_freq is not None:
            if pass_zero == 'bandpass':
                print("Warning: Changing pass_zero from 'bandpass' to 'lowpass' for single low-pass filter.")
                pz = 'lowpass'
            else:
                pz = pass_zero
            taps = signal.firwin(numtaps=numtaps_lp, cutoff=h_freq, pass_zero=pz, window='hamming', fs=sfreq)
    
    # Effective delay
    numtaps = len(taps)
    delay = int((numtaps - 1) // 2)
    slices = [slice(None)] * number_of_dims

    if call_type == "both":
        filtered = signal.filtfilt(b=taps, a=1.0, x=array, axis=axis)

    # Forward use zero-phase filtering with initial conditions
    if call_type in ["forward", "forward_compensated_cut"]:
        zi = signal.lfilter_zi(taps, 1.0)
        zi_view_shape = [1] * number_of_dims
        zi_view_shape[axis] = len(zi) 
        zi_expanded = zi.reshape(zi_view_shape)
        
        slice_idx = [slice(None)] * number_of_dims
        slice_idx[axis] = slice(0, 1) 
        x0 = array[tuple(slice_idx)]
        zi_shaped = zi_expanded * x0 
        
        filtered_raw, _ = signal.lfilter(b=taps, a=1.0, x=array, axis=axis, zi=zi_shaped)
        
        # Compensate for delay by cutting initial samples
        if call_type == "forward_compensated_cut":
            slices[axis] = slice(None,-delay)
            filtered =  filtered_raw[tuple(slices)]
        # No compensation --> additional phase delay 
        else:
            filtered = filtered_raw

    # Reflected mode (Uses FFT for extreme speed offline)
    elif call_type == "forward_compensated_reflected":
        pad_len = numtaps - 1
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (pad_len, pad_len)
        array_padded = np.pad(array, pad_width, mode='reflect')
        
        if use_fourier:
            # Reshape for correct broadcasting
            shape_taps = [1] * array.ndim
            shape_taps[axis] = -1
            taps_reshaped = taps.reshape(shape_taps)
            
            # FFT Convolution
            filtered_full = signal.convolve(array_padded, taps_reshaped, mode='full', method='auto')
            
            # Slice to maintain size consistent with lfilter
            slices_full = [slice(None)] * array.ndim
            slices_full[axis] = slice(0, array_padded.shape[axis])
            filtered_padded = filtered_full[tuple(slices_full)]
        else:
            filtered_padded, _ = signal.lfilter(b=taps, a=1.0, x=array_padded, axis=axis)

        # Final zero-phase cut
        start = delay + pad_len
        stop = start + array.shape[axis]
        slices[axis] = slice(start, stop)

        # Return filtered data --> there is no delay to compensate here because of the reflection padding (distorsion at edges is avoided)
        filtered = filtered_padded[tuple(slices)]
        
    if l_freq is None:
        filtered += dc_offset
    else:
        # The DC offset has been removed by the high-pass filter
        pass
    if return_delay:
        return delay, filtered
    else:
        return filtered

# ===================
# TRF MODEL UTILITIES
def _compute_shifted(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]]
) -> torch.Tensor:
    """
    Compute shifted matrix for given features and delays.

    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : Sequence[int]
        Delays to apply to the features.
    indices_to_keep : Optional[Sequence[int]]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    # Convert delays to tensor only once
    if not isinstance(delays, torch.Tensor):
        delays = torch.tensor(delays, device=device, dtype=torch.int64)

    if indices_to_keep is not None:
        # Convert indices to tensor only if not already a tensor
        if not isinstance(indices_to_keep, torch.Tensor):
            idx = torch.tensor(indices_to_keep, device=device, dtype=torch.int64)
        else:
            idx = indices_to_keep.to(device=device, dtype=torch.int64)
        n_rows = idx.shape[0]
        idx_shifted = idx.unsqueeze(1) - delays.unsqueeze(0)  # More explicit broadcasting
    else:
        n_rows = n_samples
        idx_shifted = torch.arange(n_samples, device=device, dtype=torch.int64).unsqueeze(1) - delays.unsqueeze(0)

    # Mask for valid indices - combine operations
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)

    # Pre-allocate output tensor with correct shape
    feats_exp = torch.zeros((n_rows, delays.shape[0], n_features), 
                           dtype=feats_t.dtype, device=device)
    
    # Only process valid indices to avoid unnecessary operations
    if valid_mask.any():
        # Clamp and gather only valid indices
        idx_clipped = idx_shifted.clamp(0, n_samples - 1)
        
        # Use advanced indexing more efficiently
        feats_gathered = feats_t[idx_clipped]  # Shape: (n_rows, n_delays, n_features)
        
        # Apply mask in-place to avoid extra memory allocation
        feats_gathered.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
        feats_exp = feats_gathered

    return feats_exp

def _compute_shifted_optimized(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """
    Optimized for medium-sized delay arrays (~100 delays).
    Uses memory-efficient chunking with vectorized operations.
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    # Convert to tensor once
    if not isinstance(delays, torch.Tensor):
        delays = torch.tensor(delays, device=device, dtype=torch.int64)
    
    if indices_to_keep is not None:
        if not isinstance(indices_to_keep, torch.Tensor):
            idx = torch.tensor(indices_to_keep, device=device, dtype=torch.int64)
        else:
            idx = indices_to_keep.to(device=device, dtype=torch.int64)
        n_rows = idx.shape[0]
    else:
        idx = torch.arange(n_samples, device=device, dtype=torch.int64)
        n_rows = n_samples
    
    n_delays = delays.shape[0]
    
    # Pre-allocate output
    result = torch.zeros((n_rows, n_delays, n_features), 
                        dtype=feats_t.dtype, device=device)
    
    # Process in chunks to balance memory vs speed
    chunk_size = min(32, n_delays)  # Adjust based on your GPU memory
    
    for start_delay in range(0, n_delays, chunk_size):
        end_delay = min(start_delay + chunk_size, n_delays)
        delay_chunk = delays[start_delay:end_delay]
        
        # Vectorized computation for this chunk
        idx_shifted = idx.unsqueeze(1) - delay_chunk.unsqueeze(0)
        valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)
        
        # Only process if there are valid indices
        if valid_mask.any():
            idx_clipped = idx_shifted.clamp(0, n_samples - 1)
            chunk_result = feats_t[idx_clipped]
            chunk_result.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            result[:, start_delay:end_delay, :] = chunk_result
    
    return result

def shifted_matrix(
    features: np.ndarray,
    delays: Sequence[int],
    use_gpu: bool = True,
    indices_to_keep: Optional[Sequence[int]] = None,
    output_torch: bool = False,
    train_indexes: np.ndarray = None,
    pred_indexes: np.ndarray = None,
    optimized_shifted: bool = False
) -> np.ndarray:
    """
    Build a time-shifted design matrix for given features and delays.

    This function stacks time-shifted versions of the input feature matrix along the second axis,
    optionally computing only for specified row indices to reduce memory.

    Parameters
    ----------
    features : np.ndarray, shape (n_times, n_features) or (n_times,)
        Input time series data. If 1D, it is treated as a single feature.
    delays : Sequence[int]
        Relative time shifts (in samples). Positive delays shift past values,
        negative delays shift future values, zero retains current.
    use_gpu : bool, default True
        Whether to attempt computation on CUDA device first. Falls back to CPU on OOM.
    indices_to_keep : Sequence[int], optional
        Specific time indices at which to compute rows of the shifted matrix.
        If None, computes all rows.
    output_torch : bool or float, default False
        If True, returns a PyTorch tensor instead of a NumPy array. 
        If False, returns a NumPy array.
    train_indexes : np.ndarray, optional
        Indices of training samples. If provided, only these indices are used for computation.
    pred_indexes : np.ndarray, optional
        Indices of prediction samples. If provided, only these indices are used for computation.
    optimized_shifted : bool, default False
        If True, uses optimized shifted matrix computation for large datasets.

    Returns
    -------
    np.ndarray, shape (n_rows, n_features * n_delays)
        Design matrix where each row contains concatenated features for each delay.
    """
    # Determine device order: try GPU first, then CPU
    preferred = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    devices = [preferred]
    if preferred.type == "cuda":
        devices.append(torch.device("cpu"))

    # Ensure features is 2D
    feats = features.reshape(-1, 1) if features.ndim == 1 else features
    for dev in devices:
        try:
            # Move data onto device
            feats_t = torch.tensor(feats.astype(np.float64), dtype=torch.float32, device=dev)
            if optimized_shifted:
                shifted = _compute_shifted_optimized(feats_t, delays, indices_to_keep)
            else:
                shifted = _compute_shifted(feats_t, delays, indices_to_keep)
            
            # Reshape: (n_rows, n_delays, n_features) -> (n_rows, n_features * n_delays)
            n_rows, n_delays, n_feat = shifted.shape
            mat = shifted.permute(0, 2, 1).reshape(n_rows, n_feat * n_delays)
            if train_indexes is not None and pred_indexes is not None:
                if output_torch:
                    return mat[train_indexes, :], mat[pred_indexes, :]
                else: 
                    return mat[train_indexes, :].cpu().numpy(), mat[pred_indexes, :].cpu().numpy()
            else:
                if output_torch:
                    return mat
                else: 
                    return mat.cpu().numpy()

        except RuntimeError as e:
            if dev.type == "cuda":
                print(f"CUDA OOM on device {dev}; retrying on CPU. Error: {e}")
                continue
            else:
                raise
    # If loop completes without return, something went wrong
    raise RuntimeError("shifted_matrix failed on all devices")

class Standarize():
    def __init__(
        self, 
        axis:int=0, 
        by_gpu:bool=False
    )->None:
        """
        Standarize train and test data to be used in a linear regressor model. 

        Parameters
        ----------
        axis : int, optional
            Axis to perform standrize, by default 0
        by_gpu : bool, optional
            Whether to perform computation on GPU, by default False
        """
        self.axis = axis
        self.by_gpu = by_gpu
        self.device = torch.device("cuda" if by_gpu and torch.cuda.is_available() else "cpu")

    def _to_device(
        self, 
        data:np.ndarray
        )->torch.Tensor:
        """
        Move data to GPU if by_gpu is True, and ensure dtype is float32.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be moved to GPU if by_gpu is True.
            
        Returns
        -------
        torch.Tensor
            Data moved to GPU if by_gpu is True, in float32.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            if (data.dtype != torch.float32):
                data = data.to(dtype=torch.float32)
            else:
                pass
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

        if self.by_gpu and torch.cuda.is_available():
            return data.to(self.device)
        else:
            return data

    def fit_standarize_train(
        self, 
        train_data:np.ndarray
        )->np.ndarray:
        """
        Standardize train data, also define mean and std to standardize future data.
        
        Parameters
        ----------
        train_data : np.ndarray
            Train data to be standardized.
            
        Returns
        -------
        np.ndarray
            Standardized train data.
        """
        train_data = self._to_device(train_data)
        
        # Fix mean and standard deviation with train data
        if isinstance(train_data, torch.Tensor):
            self.mean = train_data.mean(dim=self.axis,  keepdim=True)
            self.std = train_data.std(dim=self.axis, unbiased=False,  keepdim=True)  # Use biased std for consistency with numpy
        else:
            self.mean = train_data.mean(axis=self.axis)
            self.std = train_data.std(axis=self.axis)   

        # Standardize data
        train_data -= self.mean
        train_data /= (self.std + 1e-12)  # Adding epsilon to avoid division by zero
        
        return train_data

    def fit_standarize_test(
        self, 
        test_data:np.ndarray
        )->np.ndarray:
        """
        Standardize test data with mean and std of train data
        
        Parameters
        ----------
        test_data : np.ndarray

        Returns
        -------
        np.ndarray
            Standardized test data.
        """
        test_data = self._to_device(test_data)
        
        # Standardize with mean and standard deviation of train
        test_data -= self.mean
        test_data /= (self.std + 1e-12)
        
        return test_data

    def standarize_data(
        self,
        data:np.ndarray
        )->np.ndarray:
        """
        Standardize data with its own mean and standard deviation.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be standardized.
        
        Returns
        -------
        np.ndarray
            Standardized data.
        """
        data = self._to_device(data)
        
        if isinstance(data, torch.Tensor):
            data -= data.mean(dim=self.axis, keepdim=True)
            data /= data.std(dim=self.axis,  keepdim=True, unbiased=False)  # Use biased std for consistency with numpy
        else:
            data -= data.mean(axis=self.axis)
            data /= data.std(axis=self.axis)
            
        return data
    
class Normalize():
    def __init__(
        self, 
        axis:int=0, 
        porcent:float=5, 
        by_gpu:bool=False
        )->None:
        """
        Normalize train and test data to be used in a linear regressor model.

        Parameters
        ----------
        axis : int, optional
            Axis to perform normalize, by default 0
        porcent : float, optional
            Percentage for normalization, by default 5
        by_gpu : bool, optional
            Whether to perform computation on GPU, by default False
        """
        self.axis = axis
        self.porcent = porcent
        self.by_gpu = by_gpu
        self.device = torch.device("cuda" if by_gpu and torch.cuda.is_available() else "cpu")

    def _to_device(
        self, 
        data:np.ndarray
        )->torch.Tensor:
        """
        Move data to GPU if by_gpu is True, and ensure dtype is float32.
        
        Parameters
        ----------
        data : np.ndarray
            Data to be moved to GPU if by_gpu is True.
            
        Returns
        -------
        torch.Tensor
            Data moved to GPU if by_gpu is True, in float32.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            if (data.dtype != torch.float32):
                data = data.to(dtype=torch.float32)
            else:
                pass
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

        if self.by_gpu and torch.cuda.is_available():
            return data.to(self.device)
        else:
            return data

    def fit_normalize_train(
        self, 
        train_data:np.ndarray
        )->np.ndarray:
        """
        Normalize train data, also define min and max to normalize future data
        
        Parameters
        ----------
        train_data : np.ndarray
            Train data to be normalized.
            
        Returns
        -------
        np.ndarray
            Normalized train data.
        """
        train_data = self._to_device(train_data)
        
        # Remove offset by minimum
        if isinstance(train_data, torch.Tensor):
            self.min = train_data.min(dim=self.axis)[0]
        else:
            self.min = train_data.min(axis=self.axis)
        train_data -= self.min

        # Normalize by maximum
        if isinstance(train_data, torch.Tensor):
            self.max = train_data.max(dim=self.axis)[0]
        else:
            self.max = train_data.max(axis=self.axis)
            
        train_data = train_data / (self.max + 1e-12)  # Adding epsilon to avoid division by zero
        
        return train_data

    def fit_normalize_test(
        self, 
        test_data:np.ndarray
        )->np.ndarray:
        """
        Normalize test data with min and max of train data.
        
        Parameters
        ----------
        test_data : np.ndarray
            Test data to be normalized.
        
        Returns
        -------
        np.ndarray
            Normalized test data.
        """
        test_data = self._to_device(test_data)
        
        test_data -= self.min
        test_data = test_data / (self.max + 1e-12)
        
        return test_data

    def normalize_data(
        self, 
        data:np.ndarray, 
        kind:str="1"
        )->np.ndarray:
        """
        Normalize data
        
        Parameters
        ----------
        data : np.ndarray
            Data to be normalized.
        kind : str, optional
            Type of normalization, by default '1'
        
        Returns
        -------
        np.ndarray
            Normalized data.
        """
        data = self._to_device(data)
        
        if isinstance(data, torch.Tensor):
            data -= data.min(dim=self.axis)[0]
            data /= data.max(dim=self.axis)[0]
        else:
            data -= data.min(axis=self.axis)
            data /= data.max(axis=self.axis)
        
        if kind == '2':
            data *= 2
            data -= 1
        
        return data

    def fit_normalize_percent(
        self, 
        data:np.ndarray
        )->np.ndarray:
        """
        Normalize data using percentiles
        
        Parameters
        ----------
        data : np.ndarray
            Data to be normalized.
        
        Returns
        -------
        np.ndarray
            Normalized data.
        """
        data = self._to_device(data)
        
        # Calculate n
        n = int((self.porcent * len(data) - 1) / 100) 
        
        # Find the n-th minimum and offset that value
        sorted_data = copy.deepcopy(data)
        sorted_data.sort(self.axis)
        min_data_n = sorted_data[n]
        data -= min_data_n

        # Find the n-th maximum
        sorted_data = copy.deepcopy(data)
        sorted_data.sort(self.axis)
        max_data_n = sorted_data[-n]
        
        # Normalize data
        data = data / (max_data_n + 1e-12)  # Adding epsilon to avoid division by zero
        
        return data

# =================
# EEG MISCELLANEOUS
def band_selection(
    band: str
) -> tuple[Union[float, None], Union[float, None]]:
    """
    Select frequency band limits based on predefined bands.

    Parameters
    ----------
    band : str
        The name of the frequency band. Options are:
        - 'Delta': 1-4 Hz
        - 'Theta': 4-8 Hz
        - 'Alpha': 8-12 Hz
        - 'Beta': 12-30 Hz
        - 'Gamma': 30-100 Hz
        - 'Broad': 1-15 Hz
        - 'FullBand': None-None (no filtering)

    Returns
    -------
    tuple[Union[float, None], Union[float, None]]
        The low and high frequency limits for the selected band.
    """
    bands = {
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (12.0, 30.0),
        'Broad': (1.0, 15.0),
        'FullBand': (None, None)
    }
    if band not in bands:
        raise ValueError(f"Band '{band}' is not recognized. Available bands: {list(bands.keys())}")
    
    return bands[band]

def get_info_mne(
    montage_name: str = 'biosemi64',
    ch_types: Union[str, list] = 'eeg',
    channel_selection: Union[list, None] = None,
    sample_frequency: int = 128,
    return_channels_index: bool = False
) -> mne.Info:
    """
    Extract basic information from an MNE Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.

    Returns
    -------
    mne.Info
        Information dictionary containing sampling frequency, number of channels, and channel names.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    if channel_selection is None:
        channel_selection = montage.ch_names
    info = mne.create_info(
        ch_names=channel_selection,
        sfreq=sample_frequency,
        ch_types=ch_types
    )
    info.set_montage(montage)
    if return_channels_index:
        channels_index = [info.ch_names.index(ch) for ch in channel_selection]
        return channels_index, info
    else:
        return info

# =====================
# ANNOTATION PROCESSING
def find_first_event_on_id(
        raw, 
        trigger_signal, 
        event_id
    ) -> np.ndarray:
        """
        Find the time of the first occurrence of a specific event ID in the trigger signal.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw EEG data.
        trigger_signal : np.ndarray
            The trigger signal array.
        event_id : int
            The event ID to search for.
        
        Returns
        -------
        np.ndarray
            The times of the first occurrence of the specified event ID.
        """
        return raw.times[
            np.where(
                np.diff(
                    (trigger_signal==event_id).astype(int)
                )==1
            )[0]+1
        ]
        # return np.where(
        #             (trigger_signal==event_id).astype(int)
        #     )[0]+1

def get_no_task_times(
    raw: mne.io.Raw,
    onsets: np.ndarray,
    offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get onsets and durations of no-task periods based on task period onsets and offsets.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    onsets : np.ndarray
        Array of task period onset times.
    offsets : np.ndarray
        Array of task period offset times.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Onsets and durations of no-task periods.
    """
    no_task_mask = np.zeros_like(raw.times, dtype=bool)

    # Mark periods between offsets and next onsets as no-task
    for onset, offset in zip(onsets, offsets):
        onset_idx = raw.time_as_index(onset)[0]
        offset_idx = raw.time_as_index(offset)[0]
        no_task_mask[onset_idx:offset_idx] = True
    no_task_mask = ~no_task_mask

    diff = np.diff(no_task_mask.astype(int),  append=0)
    diff[0] = no_task_mask[0]
    offsets_not_annotated = raw.times[np.where(diff==-1)[0]]
    onsets_not_annotated = raw.times[np.where(diff==1)[0]]
    durations_not_annotated = offsets_not_annotated - onsets_not_annotated

    return onsets_not_annotated, durations_not_annotated

