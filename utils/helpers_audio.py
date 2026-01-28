"""
This module provides helper functions for audio processing tasks such as reading,
extracting envelopes, spectrograms, and phonemes, as well as computing stimulus
representations from audio data.
"""
from typing import Dict, Union
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
import librosa

from phonet.phonet import Phonet  # type: ignore --> to supress pylance problem

from utils.helpers_processing import (
    custom_resample
)
import config
np.random.seed(config.RANDOM_SEED)

LEXICON = {
    'phones': [
        '<p:>', 'B', 'D', 'F', 'G', 'J', 'L', 'N', 'S', 'T', 'Z', 'a', 'b', 'd', 'e',
        'f', 'g', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 'sil',
        't', 'tS', 'u', 'w', 'x', 'z'
    ],
    'phonemes': [
            '/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/',
            '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/','/sil/'
        ],
    'phones_to_phonemes': {
            'a' : '/a/', 'e' : '/e/', 'i' : '/i/', 'o' : '/o/', 'j' : '/i/', 'w' : '/u/', 'u' : '/u/',
            'l' : '/l/', 'r' : '/R/', 'rr': '/r/', 't' : '/t/', 'd' : '/d/', 'D' : '/d/', 'sil' : '/sil/',
            '<p:>' : '/sil/', 'm' : '/m/', 'n' : '/n/', 'N' : '/n/', 'k' : '/k/', 'g' : '/g/', 'G' : '/g/',
            'tS': '/tS/', 'T' : '/tS/', 'f' : '/f/', 'F' : '/f/', 's' : '/s/', 'S' : '/s/', 'z' : '/s/',
            'Z' : '/s/', 'p' : '/p/', 'b' : '/b/', 'B' : '/b/', 'L' : '/L/', 'x' : '/x/', 'jj': '/x/', 'J' : '/x/'
        },
    'phonological_labels': {
            "vocalic" : ["a","e","i","o","u", "w", "j"],
            "consonantal" : ["b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z"],
            "back" : ["a","o","u", "w"],
            "anterior" : ["e","i","j"],
            "open" : ["a","e","o"],
            "close" : ["j","i","u", "w"],
            "nasal" : ["m","n", "N"],
            "stop" : ["p","b", "B","t","k","g", "G","tS","d", "D"],
            "continuant" : ["f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z"],
            "lateral" :["l"],
            "flap" :["r"],
            "trill" :["rr"],
            "voice" :["a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j"],
            "strident" :["tS","f", "F","s", "Z", "T", "z",  "S"],
            "labial" :["m","p","b", "B","f", "F"],
            "dental" :["t","d", "D"],
            "velar" :["k","g", "G"],
            "pause" :  ["sil", "<p:>"]
        },
    'phonological_labels1': ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal'],
    'phonological_labels2': ['dental', 'consonantal', 'velar', 'flap', 'close', 'strident', 'continuant']
}

def _get_phonet_instance(
    phonet_cache: dict = {},
    logger:logging.Logger=None,
):
    """
    Get a cached Phonet instance to avoid reloading models
    """
    if 'phonet' not in phonet_cache:
        logger.debug("Loading Phonet model (first time only)...") if logger else None
        phonet_cache['phonet'] = Phonet(["all"])
        logger.debug("Phonet model loaded and cached") if logger else None
    
    return phonet_cache

def _get_dnn_instance(
    dnn_cache: dict = {},
    model: str = 'base',
    logger:logging.Logger=None,
):
    """
    Get a cached DNN instance to avoid reloading models
    """
    if 'dnn' not in dnn_cache:
        logger.debug("Loading DNN model (first time only)...") if logger else None
        if model not in dnn_cache:
            import whisper
            dnn_cache[model] = whisper.load_model(model)
        logger.debug("DNN model loaded and cached") if logger else None
    
    return dnn_cache

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

def save_wav(
    file_path: Union[str, Path],
    sample_rate: int,
    data: np.ndarray
) -> None:
    """
    Saves a numpy array as a WAV file.
    
    Parameters
    ----------
        file_path: Union[str, Path]
            Path to save the WAV file
        sample_rate: int
            Sample rate of the audio
        data: np.ndarray
            Audio data as a numpy array
    
    Returns
    -------
        None
    """
    wavfile.write(file_path, sample_rate, data)
def read_wav(
    file_path: Union[str, Path],
    return_sample_rate: bool = False
) -> Union[np.ndarray, tuple[int, np.ndarray]]:
    """
    Reads a WAV file and returns it as an array.
    
    Parameters
    ----------
        file_path: Union[str, Path]
            Path to the WAV file
        return_sample_rate: bool
            If True, returns a tuple (sample_rate, data)
    
    Returns
    -------
        np.ndarray or tuple[int, np.ndarray]: Audio data as a numpy array,
            or (sample_rate, data) if return_sample_rate is True
    """
    sample_rate, data = wavfile.read(file_path)
    if return_sample_rate:
        return sample_rate, data
    else:
        return data

def transcribe_audio(
    audio_filepath: Union[str, Path],
    dnn_model: str = "base",
    dnn_cache: dict = {},
    timestamp: bool = False,
    language: str = "es"
) -> Union[str, list]:
    """
    Transcribes audio using OpenAI's Whisper model.
    
    Parameters
    ----------
    audio_filepath : Union[str, Path]
        Path to the audio file to transcribe.
    dnn_model : str
        The Whisper model to use for transcription.
    dnn_cache : dict
        Cache dictionary to store loaded models.
    timestamp : bool
        Whether to include word timestamps in the transcription.
    language : str
        Language code for transcription (default is "es" for Spanish).

    Returns
    -------
    Union[str, list]
        The transcribed text or segments with timestamps.
    """
    if dnn_model not in dnn_cache:
        dnn_cache = _get_dnn_instance(dnn_cache, model=dnn_model)[dnn_model]
    model = dnn_cache[dnn_model]
    result = model.transcribe(
        str(audio_filepath), 
        language=language,
        word_timestamps=timestamp
    )
    if timestamp:
        transcription = result['segments']
        return transcription
    else:
        transcription = result['text']
        return transcription

def compute_phones(
    phonet_obj,
    audio_signal: np.ndarray,
    sample_rate: int,
    PLLR:bool=False,
    target_sr:int=config.TARGET_SAMPLING_RATE
)-> np.ndarray:
    """
    Compute phones from the audio file.
    
    Parameters
    ----------
    phonet_obj : Phonet
        An instance of the Phonet class for phoneme extraction.
    audio_signal: np.ndarray
        Audio signal array.
    sample_rate: int
        Sampling rate of the audio signal.
    target_sr : int
        Target sampling rate for the output phoneme probabilities.
    PLLR : bool
        Whether to return the PLLR (Phoneme Loglikelihood ratio). By default, True
    
    Returns
    -------
    np.ndarray
        If PLLR is True, returns the posterior probabilities of phonemes.
        If PLLR is False, returns the discrete phoneme sequence.
    """
    # Resample to 16 kHz as required by Phonet
    resampled_sample_rate = 16000
    audio_signal = custom_resample(
        array=audio_signal,
        original_sr=sample_rate,
        target_sr=resampled_sample_rate,
        axis=0
    )
    
    # This method extracts log-Mel-filterbank energies used as inputs of the model. 
    # The output frequency is 100 Hz, which is the same as the time shift of the model.
    log_mel_filt_bank = phonet_obj.get_feat(audio_signal, resampled_sample_rate)
    
    # Calculate the number of frames represented in the audio signal
    number_of_frames = int(
        log_mel_filt_bank.shape[0]/phonet_obj.len_seq # len_seq=40 always
    ) 
    
    # Segment the mels into sequences of len_seq frames
    input_features = []
    start, end = 0, phonet_obj.len_seq
    for j in range(number_of_frames):
        input_features.append(log_mel_filt_bank[start:end,:])
        start += phonet_obj.len_seq
        end += phonet_obj.len_seq

    # Standarize the input features
    input_features = np.stack(input_features, axis=0)
    input_features = input_features-phonet_obj.MU
    input_features = input_features/phonet_obj.STD
    
    # Get the predictions from the model and concatenate them to get a sequence
    probabilities = np.asarray(
        phonet_obj.model_phon.predict(input_features)
        )
    posterior_gram = np.concatenate(
        probabilities, 
        axis=0
    )
    
    # time_shift is the time interval between frames
    total_audio_frames = int(len(audio_signal)/(phonet_obj.time_shift*resampled_sample_rate))
    posterior_gram = posterior_gram[:total_audio_frames]
    
    # posterior_prob: (num_frames, num_phones), original_fs ≈ 100 Hz
    num_target_frames = int((audio_signal.shape[0] / resampled_sample_rate) * target_sr)
    posterior_gram = signal.resample(
        posterior_gram, 
        num_target_frames, 
        axis=0
    )
    
    # sample_rate_posteriors = len(posterior_prob)/(wav.shape[0]/self.audio_sr)
    # import resampy

    # posterior_prob2 = resampy.resample(
    #     posterior_prob, 
    #     sr_orig=sample_rate_posteriors, 
    #     sr_new=config.sr, 
    #     axis=0
    # )
    
    if PLLR:
        return posterior_gram
    else:
        greedy_prediction = np.argmax(
            posterior_gram, 
            axis=1
        )
        phone_sequence = [
            str(phonet_obj.phonemes[j])
            for j in greedy_prediction
        ]
        return phone_sequence
    
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
    audio_signal: np.ndarray,
    sample_rate: int,
    envelope_length: int,
    lexicon: dict = LEXICON,
    phonet_obj: Phonet = None,
    use_unprobed_audio: bool = False,
    audio_filepath: Union[str, Path] = None,
    side: str = 'mono'
) -> np.ndarray:
    def extract_phoneme_single_side(
        audio_signal: np.ndarray,
        sample_rate: int,
        envelope_length: int,
        lexicon: dict,
        phonet_obj: Phonet
    ) -> np.ndarray:
        posterior_prob = compute_phones(
            audio_signal=audio_signal,
            sample_rate=sample_rate,
            phonet_obj=phonet_obj, 
            PLLR=True
        )
        posterior_prob = np.clip(
            posterior_prob, 1e-6, 1-1e-6
        )
        
        # Repeat last sample (probably silence)
        difference = len(posterior_prob) - envelope_length
        if difference > 0:
            posterior_prob = posterior_prob[:-difference]
        elif difference < 0:
            for i in range(np.abs(difference)):
                aux = posterior_prob[-1].copy() 
                posterior_prob = np.vstack((posterior_prob, aux.reshape(-1,1).T))
        
        # Map phones to phonemes, making the sum
        posterior_prob_phonemes = np.zeros(
            shape=(posterior_prob.shape[0], len(lexicon['phonemes']))
        )
        for h, phone in enumerate(lexicon['phones']):
            phoneme_index = lexicon['phonemes'].index(
                lexicon['phones_to_phonemes'][phone]
            )
            posterior_prob_phonemes[:, phoneme_index] += posterior_prob[:, h]
                    
        # Calculate posterior llr
        pllr = np.zeros(shape=posterior_prob_phonemes.shape)
        number_of_phonemes = posterior_prob_phonemes.shape[1]
        for ph in range(number_of_phonemes):
            if np.isinf(posterior_prob_phonemes[:, ph]/(1-posterior_prob_phonemes[:, ph]+ 1e-8)).any():
                print("There are infinite values in pllr")
            safe_vals = np.clip(
                posterior_prob_phonemes[:, ph] / (1 - posterior_prob_phonemes[:, ph] + 1e-8), 
                1e-8, None
            )
            pllr[:, ph] = np.log10(
                safe_vals
            )

        # Centralizamos 
        pllr = np.nan_to_num(pllr, nan=0.0, posinf=0.0, neginf=0.0)
        pllr = pllr - np.mean(pllr, axis=1, keepdims=True)  
        
        # Removemos silencios
        pllr_without_silence = pllr[:, np.arange(number_of_phonemes) != lexicon['phonemes'].index('/sil/')]
        return pllr_without_silence
    
    if use_unprobed_audio:
        if audio_filepath is None:
            raise ValueError("audio_filepath must be provided when use_unprobed_audio is True.")
        audio_stereo, sample_rate = librosa.load(
            path=str(audio_filepath).replace('with_probe', 'no_probe').replace('tone', 'no'),
            sr=None,
            mono=False
        )
        if side == 'left':
            audio_stereo = audio_stereo.T[:, 0]
        elif side == 'right':
            audio_stereo = audio_stereo.T[:, 1]
        else:
            audio_stereo = audio_stereo.T
    else:
        audio_stereo = audio_signal
    if audio_stereo.ndim > 1:
        pllrs_sides = []
        for audio_signal in [audio_stereo[:, 0], audio_stereo[:, 1]]:
            pllrs_sides.append(
                extract_phoneme_single_side(
                    audio_signal=audio_signal,
                    sample_rate=sample_rate,
                    envelope_length=envelope_length,
                    lexicon=lexicon,
                    phonet_obj=phonet_obj
                )
            )
        return np.mean(pllrs_sides, axis=0)
    else:
        pllrs = extract_phoneme_single_side(
            audio_signal=audio_signal,
            sample_rate=sample_rate,
            envelope_length=envelope_length,
            lexicon=lexicon,
            phonet_obj=phonet_obj
        )
        return pllrs

def extract_phonemes_discrete(
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
    params: Dict = {},
    side: str = "mono",
    logger: logging.Logger = None
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
    side : str, optional
        The side of the audio channel ('left', 'right' or 'mono'). Default is None.
    
    Returns
    -------
        np.ndarray
            The computed stimulus representation.
    """
    POSSIBLE_STIMULI = ['Envelope', 'Spectrogram', 'BipOnsets', 'Phonemes', 'PhonemesDiscrete']
    
    if attribute not in POSSIBLE_STIMULI:
        raise ValueError(f"Attribute '{attribute}' not recognized. Must be one of {POSSIBLE_STIMULI}.")
    if attribute == 'Envelope':
        audio = np.mean(audio, axis=1) if side == 'mono' else audio
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
        audio = np.mean(audio, axis=1) if side == 'mono' else audio
        
        # Then extract spectrogram is already build to resample inside
        stimulus_resampled = extract_spectrogram(
            audio_data=audio,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            **params
        )
    elif attribute == 'BipOnsets':
        audio = np.mean(audio, axis=1) if side == 'mono' else audio
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
        stimulus_resampled = extract_phonemes(
            audio_signal=audio,
            sample_rate=sample_rate,
            audio_filepath=audio_filepath,
            side=side,
            **params
        )
    elif attribute == 'PhonemesDiscrete':
        stimulus = extract_phonemes_discrete(
            audio_signal=audio,
            sample_rate=sample_rate,
            side=side,
            **params
        )
    # Compensate for delay
    if delay_to_compensate is not None and delay_to_compensate > 0:
        stimulus_resampled = stimulus_resampled[:-delay_to_compensate]
    elif delay_to_compensate is not None and delay_to_compensate < 0:
        raise NotImplementedError("Negative delay doesn't make sense.")
    return stimulus_resampled