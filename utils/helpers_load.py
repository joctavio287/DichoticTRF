from typing import Dict, List, Union, Tuple
from collections.abc import Iterable
from pathlib import Path
import numpy as np

from utils.helpers_audio import (
    compute_single_stimulus
)
import config

    
def extract_stimuli(
    audios: Union[np.ndarray, Tuple[np.ndarray, ...]],
    sample_rate: int,
    target_sample_rate: int,
    delay_to_compensate: int,
    attributes: List[str] = ['Envelope', 'Spectrogram', 'Phonemes'],
    attribute_params: Dict[str, Dict] = None,
    axis: int = 0,
    subj_dir: Union[Path, str, None] = None,
    overwrite: bool = True,
    tag: str = "",
    side: Union[str, None] = None
) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    Extracts stimulus representations from audio data. 
    If iterable, processes each audio channel separately. In this case, If `subj_dir` 
    is provided, attempts to load existing stimulus files from disk.
    If not iterable, processes the single audio input directly without saving/loading from disk.    
    
    Parameters
    ----------
    audios : Union[np.ndarray, Tuple[np.ndarray, ...]]
        Audio data array(s) from which to extract stimulus representations. If stereo audio is provided,
        it should be a tuple of two numpy arrays (left and right channels).
    sample_rate : int
        Sampling rate of the audio data.
    target_sample_rate : int
        Target sampling rate for the extracted stimulus representations.
    delay_to_compensate : int
        Delay (in samples) to compensate for when extracting stimulus representations.
    attributes : List[str], optional
        List of stimulus attributes to extract. Default is ['Envelope', 'Spectrogram', 'Phonemes'].
    attribute_params : Dict[str, Dict], optional
        Dictionary containing parameters for each attribute extraction method. Default is an empty dictionary.
    axis : int, optional
        Axis along which to compute the stimulus representations. Default is 0.
    subj_dir : Union[Path, str, None], optional
        Directory path to save/load the extracted stimulus representations. If None, stimuli will not be saved/loaded.
        Default is None.
    overwrite : bool, optional
        If True, overwrites existing stimulus files in `subj_dir`. Default is True.
    tag : str, optional
        A tag to append to the stimulus filenames when saving/loading. Default is an empty string.
    side : Union[str, None], optional
        The side of the audio channel to process ('left', 'right', 'mono' or 'both'). Default is None.
    
    Returns
    -------
    Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]
        A dictionary where keys are attribute names and values are the corresponding extracted representations
        keeping the same structure as the input audios.
    """
    attribute_params = {} if attribute_params is None else attribute_params
    def load_single_stimuli(
        audio_: np.ndarray,  
        attribute_: str,
        params_: Dict,
        sample_rate_: int,
        target_sample_rate_: int,
        axis_: int = 0,
        delay_to_compensate_: Union[int, None] = None,
        stimulus_path_: Union[Path, None] = None,
        overwrite_: bool = False
    ) -> np.ndarray:
        stimulus_ = None
        # Try to load existing stimulus representation
        if stimulus_path_ is not None and not overwrite_:
            try:
                data = np.load(stimulus_path_, allow_pickle=True)
                stimulus_ = data['array']   
            except (FileNotFoundError, ValueError, OSError):
                stimulus_ = None
        if stimulus_ is None:   
            stimulus_ = compute_single_stimulus(
                audio=audio_,
                attribute=attribute_,
                sample_rate=sample_rate_,
                target_sample_rate=target_sample_rate_,
                axis=axis_,
                delay_to_compensate=delay_to_compensate_,
                params=params_
            )
            if stimulus_path_ is not None:
                np.savez(
                    file=stimulus_path_, 
                    array=stimulus_,
                    params=params_,
                    sample_rate=target_sample_rate
                )
        return stimulus_
    assert side in ['left', 'right', 'mono', 'both', None], "side parameter must be 'left', 'right', 'mono', 'both', or None."
    is_iterable = isinstance(audios, Iterable) and not isinstance(audios, (np.ndarray, str, bytes))
    
    if is_iterable:
        stimuli = {attr: [] for attr in attributes}
        if side == 'both':
            tag_fn = lambda j: ['left', 'right'][j] + f"_{tag}.npz"
        elif side == 'mono':
            tag_fn = lambda j: f"{side}_{tag}.npz"
            audios = (np.mean(audios, axis=axis),)
        elif side is None:
            tag_fn = None  # no disk I/O
        else:
            raise ValueError("If audios is iterable, side must be 'both' or None.")
        for j, audio in enumerate(audios):
            for attr in attributes:
                params = attribute_params.get(attr, {})
                if subj_dir is None or tag_fn is None:
                    print( "Warning: subj_dir is None, so stimuli will not be saved/loaded from disk.")
                    stimulus_path = None
                else:
                    stimulus_path = Path(subj_dir) / f"{attr.lower()}" / tag_fn(j)
                    stimulus_path.parent.mkdir(parents=True, exist_ok=True)
                stimulus = load_single_stimuli(
                    audio_=audio,
                    attribute_=attr,
                    params_=params,
                    axis_=axis,
                    sample_rate_=sample_rate,
                    target_sample_rate_=target_sample_rate,
                    delay_to_compensate_=delay_to_compensate,
                    stimulus_path_=stimulus_path,
                    overwrite_=overwrite
                )
                stimuli[attr].append(stimulus)
        # Convert lists to tuples for immutability
        for attr in stimuli:
            if side == 'mono':
                stimuli[attr] = stimuli[attr][0]
            else:
                stimuli[attr] = tuple(stimuli[attr])
    else:
        stimuli = {}
        if side not in ['left', 'right', None]:
            raise ValueError("If audios is not iterable, side must be 'left', 'right' or None.")
        tag_ = f"{side}_{tag}.npz" if side is not None else None
        for attr in attributes:
            params = attribute_params.get(attr, {})
            if subj_dir is None or tag_ is None:
                print( "Warning: subj_dir is None, so stimuli will not be saved/loaded from disk.")
                stimulus_path = None
            else:
                stimulus_path = Path(subj_dir) / f"{attr.lower()}" / tag_
                stimulus_path.parent.mkdir(parents=True, exist_ok=True)
            stimuli[attr] = load_single_stimuli(
                audio_=audios,
                attribute_=attr,
                params_=params,
                axis_=axis,
                sample_rate_=sample_rate,
                target_sample_rate_=target_sample_rate,
                delay_to_compensate_=delay_to_compensate,
                stimulus_path_=stimulus_path,
                overwrite_=overwrite
            )
    return stimuli

    
def listening_data(
    participant_id: str,
    band_freq: str,
    target_sample_rate: int,
    attributes: List[str]=['Envelope'],
    attribute_params: Dict[str, Dict]=None,
    overwrite: bool = False,
    side: str = 'both'
)-> Union[tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]], tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Loads the preprocessed EEG and attributes for a given participant during the listening task.
    
    Parameters
    ----------
    participant_id : str
        The identifier of the participant whose data is to be loaded.
    band_freq : str
        The frequency band used during preprocessing (e.g., 'delta', 'theta').
    target_sample_rate : int
        The target sample rate for the audio data.
    attributes : List[str], optional
        List of attributes to extract from the audio data. Default is ['Envelope'].
    attribute_params : Dict[str, Dict], optional
        Parameters for each attribute extraction. Default is an empty dictionary.
    overwrite : bool, optional
        If True, overwrites existing preprocessed data. Default is False.
    side : str, optional
        The side of the audio channel to process ('left', 'right', 'mono' or 'both'). Default is 'both'.
    
    """
    attribute_params = {} if attribute_params is None else attribute_params        
    # Parse all the segments for the participant
    subj_dir = config.PREPROCESSED_LISTENING_DIR  / participant_id
    eeg_subj_directory = subj_dir / band_freq
    audio_subj_directory = subj_dir / "audios"
    eeg_segments = []
    attributes_segments = []
    for n_segment, (eeg_npz_file, audio_npz_file) in enumerate(
        zip(eeg_subj_directory.glob("*.npz"), audio_subj_directory.glob("*.npz"))
    ):
        eeg_data = np.load(eeg_npz_file, allow_pickle=True)
        audio_data = np.load(audio_npz_file, allow_pickle=True)
        eeg_segments.append(eeg_data['eeg'])
        attributes_segments.append(
            extract_stimuli(
                audios=(audio_data['audio_l'], audio_data['audio_r']),
                sample_rate=int(audio_data['sfreq']),
                target_sample_rate=target_sample_rate,
                delay_to_compensate=int(eeg_data['delay_to_compensate']),
                attributes=attributes,
                attribute_params=attribute_params,
                subj_dir=subj_dir,
                tag=f"segment_{n_segment+1}",
                overwrite=overwrite,
                side=side
            )
        )
    
    # Concatenate all eeg segments
    eeg = np.concatenate(eeg_segments, axis=0)
    if side == 'both':
        # Concatenate all stimuli segments for left and right channels
        stimuli_l = {attr: [] for attr in attributes}
        stimuli_r = {attr: [] for attr in attributes}
        for segment in attributes_segments:
            for attr in attributes:
                stimuli_l[attr].append(segment[attr][0])
                stimuli_r[attr].append(segment[attr][1])
        stimuli_l = {attr: np.concatenate(stimuli_l[attr], axis=0) for attr in attributes}
        stimuli_r = {attr: np.concatenate(stimuli_r[attr], axis=0) for attr in attributes}
        
        # Match lengths (in case of small mismatches due to filtering/resampling)
        min_left = min([stimuli_l[attr].shape[0] for attr in attributes])
        min_right = min([stimuli_r[attr].shape[0] for attr in attributes])
        assert min_left == min_right, "Left and right stimuli have different minimum lengths."
        min_length = min(eeg.shape[0], min_left, min_right)
        return eeg[:min_length], ({attr: stimuli_l[attr][:min_length] for attr in attributes}, {attr: stimuli_r[attr][:min_length] for attr in attributes})
    elif side in ['mono', 'left', 'right']:
        # Concatenate all stimuli segments for mono channel
        stimuli = {attr: [] for attr in attributes}
        for segment in attributes_segments:
            for attr in attributes:
                stimuli[attr].append(segment[attr])
        stimuli = {attr: np.concatenate(stimuli[attr], axis=0) for attr in attributes}
        
        # Match lengths (in case of small mismatches due to filtering/resampling)
        min_stim = min([stimuli[attr].shape[0] for attr in attributes])
        min_length = min(eeg.shape[0], min_stim)
        return eeg[:min_length], {attr: stimuli[attr][:min_length] for attr in attributes}
    else:
        raise ValueError("side parameter must be 'both', 'mono', 'left', or 'right'.")