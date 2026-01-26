"""
This module provides helper functions for audio processing tasks such as reading,
extracting envelopes, spectrograms, and phonemes, as well as computing stimulus
representations from audio data.
"""
from typing import Dict, Union
from pathlib import Path
import pandas as pd
import numpy as np

from pydub import AudioSegment
import librosa
from scipy import signal

from utils.helpers_processing import (
    custom_resample
)
import config

np.random.seed(config.RANDOM_SEED)

def read_ogg(
    file_path: Union[str, Path],
    return_sample_rate: bool = False
) -> Union[np.ndarray, tuple[int, np.ndarray]]:
    """
    Reads an OGG file and returns it as an array.
    
    Parameters
    ----------
        file_path: Union[str, Path]
            Path to the OGG file
        return_sample_rate: bool
            If True, returns a tuple (sample_rate, data)
    
    Returns
    -------
        np.ndarray or tuple[int, np.ndarray]: Audio data as a numpy array,
            or (sample_rate, data) if return_sample_rate is True
    """
    audio = AudioSegment.from_file(file_path, format='ogg')
    sample_rate = audio.frame_rate
    data = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        data = data.reshape((-1, 2))
    if return_sample_rate:
        return sample_rate, data
    else:
        return data

def extract_envelope(
    audio_data: np.ndarray,
    axis: int = 0
)-> np.ndarray:
    """
    Compute the amplitude envelope of an audio signal using the Hilbert transform.
    
    Parameters
    ----------
        audio_data : np.ndarray
            The input audio data.
        axis : int, optional
            The axis along which to compute the envelope. Default is 0.
    
    Returns
    -------
        np.ndarray
            The amplitude envelope of the audio signal.
    """
    analytic_signal = signal.hilbert(audio_data, axis=axis)
    return np.abs(analytic_signal).reshape(-1, 1) if audio_data.ndim == 1 else np.abs(analytic_signal)

def extract_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    n_mels: int = 21
)-> np.ndarray:
    """
    Calculates spectrogram of audio data

    Parameters
    ----------
    audio_data : np.ndarray
        The input audio data.
    sample_rate : int
        The sample rate of the input audio data.
    target_sample_rate : int
        The target sampling rate for the spectrogram.
    n_mels : int
        Number of Mel bands to generate, by default 21.

    Returns
    -------
    np.ndarray
        Matrix with sprectrogram in given mel frequncies of dimension (Samples X Mel)
    """

    # Get sample window size to match the sampling rate of the EEG
    hop_length = int(np.round(sample_rate / target_sample_rate))
    
    # choose an FFT size that is >= max(hop_length, 2 * n_mels)
    def next_pow2(x):
        """ Returns the next power of 2 greater than or equal to x """
        return 1 << ((x - 1).bit_length())
    n_fft = next_pow2(max(hop_length, 2 * n_mels, 256))

    # Calculates the mel frequencies spectrogram giving the desire sampling (match the EEG)
    y = audio_data.astype(float)
    S = librosa.feature.melspectrogram(
        y=y.mean(axis=1) if y.ndim > 1 else y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
        pad_mode="reflect"
    )

    # Transform to dB using normalization to 1
    S_DB = librosa.power_to_db(
        ref=np.max,
        S=S
    ).T
    return S_DB

def extract_bip_onsets(
    audio_signal: np.ndarray,
    sample_rate: int,
    audio_filepath: Union[str, Path],
    side: str = 'mono'
) -> np.ndarray:
    """
    Extracts bip onset times from a CSV file and creates a binary mask indicating the onsets.
    
    Parameters
    ----------
    audio_signal : np.ndarray
        The input audio signal.
    sample_rate : int
        The sample rate of the audio signal.
    audio_filepath : Union[str, Path]
        The file path of the original audio file.
    side : str, optional
        The side to extract onsets for ('left', 'right', 'mono'), by default 'mono'.
    
    Returns
    -------
    np.ndarray
        A binary mask array indicating the onset times.
    """
    possible_sides = ['left', 'right', 'mono']
    if side not in possible_sides:
        raise ValueError(f"Side '{side}' not recognized. Must be one of {possible_sides}.")
    mask = np.zeros_like(audio_signal)
    audio_time = np.arange(audio_signal.shape[0]) / sample_rate
    if side == 'mono':
        onsets = pd.read_csv(
            config.ONSETS_DIR / (Path(str(audio_filepath)).stem[:7] + "_onsets.csv")
        ).to_numpy().flatten()

    else:
        onsets = pd.read_csv(
            config.ONSETS_DIR / (Path(str(audio_filepath)).stem[:7] + "_onsets.csv")
        )[side + "_s"].to_numpy()
    onset_indices = (onsets * sample_rate).astype(int)
    mask[onset_indices] = 1
    return mask.reshape(-1, 1)

def extract_phonemes(
    ):
    ...

def compute_single_stimulus(
    audio: np.ndarray, 
    attribute: str, 
    sample_rate: int, 
    axis: int = 0, 
    target_sample_rate: Union[int, None] = None, 
    delay_to_compensate: Union[int, None] = None, 
    audio_filepath: Union[str, Path] = None,
    sidename: str = 'mono',
    params: Dict = {}
) -> np.ndarray:
    """
    Computes a single stimulus representation from audio data.

    Parameters
    ----------
    audio : np.ndarray
        The input audio data.
    attribute : str
        The type of stimulus to compute ('Envelope', 'Spectrogram', 'Phonemes').
    sample_rate : int
        The sample rate of the input audio data.
    axis : int, optional
        The axis along which to compute the stimulus. Default is 0.
    target_sample_rate : Union[int, None], optional
        The target sample rate for resampling the stimulus. 
        If None, no resampling is performed. Default is None.
    delay_to_compensate : Union[int, None], optional
        Number of samples to compensate for delay. 
        If None, no delay compensation is performed. Default is None.
    params : Dict, optional
        Additional parameters specific to the stimulus extraction method.
    audio_filepath : Union[str, Path]
        The file path of the original audio file.
    sidename : str, optional
        Side to process ('left', 'right', or 'mono'). Default is 'mono'.
    
    Returns
    -------
        np.ndarray
            The computed stimulus representation.
    """
    POSSIBLE_STIMULI = ['Envelope', 'Spectrogram', 'BipOnsets', 'Phonemes']
    
    if attribute not in POSSIBLE_STIMULI:
        raise ValueError(f"Attribute '{attribute}' not recognized. Must be one of {POSSIBLE_STIMULI}.")
    if attribute == 'Envelope':
        stimulus = extract_envelope(
            audio_data=audio,
            axis=axis,
            **params
        )
        stimulus_resampled = custom_resample(
            array=stimulus,
            original_sr=sample_rate,
            target_sr=target_sample_rate,
            axis=axis
        )
    elif attribute == 'Spectrogram':
        # Then extract spectrogram is already build to resample inside
        stimulus_resampled = extract_spectrogram(
            audio_data=audio,
            sample_rate=sample_rate,
            **params
        )
    elif attribute == 'BipOnsets':
        stimulus = extract_bip_onsets(
            audio_signal=audio,
            sample_rate=sample_rate,
            audio_filepath=audio_filepath,
            **params
        )
        stimulus_resampled = custom_resample(
            array=stimulus,
            original_sr=sample_rate,
            target_sr=target_sample_rate,
            axis=axis
        )
    elif attribute == 'Phonemes':
        stimulus = extract_phonemes(
            audio_signal=audio,
            sample_rate=sample_rate,
            **params
        )

    # Compensate for delay
    if delay_to_compensate is not None and delay_to_compensate > 0:
        stimulus_resampled = stimulus_resampled[:-delay_to_compensate]
    elif delay_to_compensate is not None and delay_to_compensate < 0:
        raise NotImplementedError("Negative delay doesn't make sense.")
    return stimulus_resampled