from typing import Dict, List, Union, Tuple
from collections.abc import Iterable
from pathlib import Path
import numpy as np

from utils.helpers_processing import ( 
    load_json_to_dict
)

import config

def listening_data(
    participant_id: str,
    band_freq: str,
    attributes: List[str]=['Envelope'],
    attribute_params: Dict[str, Dict]=None,
    overwrite: bool = False,
    side: str = 'mono'
)-> Union[tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]], tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Loads the preprocessed EEG and attributes for a given participant during the listening task.
    
    Parameters
    ----------
    participant_id : str
        The identifier of the participant whose data is to be loaded.
    band_freq : str
        The frequency band used during preprocessing (e.g., 'delta', 'theta').
    attributes : List[str], optional
        List of attributes to extract from the audio data. Default is ['Envelope'].
    attribute_params : Dict[str, Dict], optional
        Parameters for each attribute extraction. Default is an empty dictionary.
    overwrite : bool, optional
        If True, overwrites existing preprocessed data. Default is False.
    side : str, optional
        The side of the audio channel to process ('left', 'right', 'mono'). Default is 'mono'.
    
    """
    attribute_params = {} if attribute_params is None else attribute_params

    # Parse all the segments for the participant
    audio_filecodes = load_json_to_dict(
        filepath=config.BEHAVIOURAL_DIR / f"{participant_id.split('prueba')[1]}_behavioural.json"
    )['metadata']['audio_filecodes']

    eeg_dir = config.PREPROCESSED_LISTENING_DIR  / 'eeg' / band_freq.lower() 
    eeg_filepaths = list(eeg_dir.glob(f"{participant_id}_segment*.npz"))
    
    attributes_segments, eeg_segments = [], []
    for eeg_npz_file, audio_filecode in zip(eeg_filepaths, audio_filecodes):
        # Load EEG segment
        eeg_data = np.load(eeg_npz_file, allow_pickle=True)
        eeg_segment = eeg_data['eeg']
        
        # Load attribute segments
        attributes_segment = {}
        for attribute in attributes:
            attribute_file = config.PREPROCESSED_LISTENING_DIR / f"{attribute.lower()}" / side / f"{audio_filecode}_{side}.npz"
            if not attribute_file.exists() or overwrite:
                from IPython import embed; embed()
                raise FileNotFoundError(f"Attribute file {attribute_file} not found. Please extract attributes first.")
            attribute_data = np.load(attribute_file, allow_pickle=True)

            # Adjust for delay compensation
            delay_to_compensate = eeg_data['delay_to_compensate']
            if delay_to_compensate == 0:
                attributes_segment[attribute] = attribute_data['attribute_values']
            else:
                attributes_segment[attribute] = attribute_data['attribute_values'][:-delay_to_compensate]
        
        # Match lengths
        min_length = min(
            eeg_segment.shape[0], 
            min(
                [attributes_segment[attr].shape[0] for attr in attributes]
            )
        )
        eeg_segments.append(eeg_segment[:min_length])
        attributes_segments.append(
            {attr: attributes_segment[attr][:min_length] for attr in attributes}
        )

    # Concatenate all eeg segments
    eeg = np.concatenate(eeg_segments, axis=0)
    try: 
        concatenated_stimuli = {attr: [] for attr in attributes}
        for segment in attributes_segments:
            for attr in attributes:
                concatenated_stimuli[attr].append(segment[attr])
        stimuli = {
            attr: np.concatenate(
                concatenated_stimuli[attr], axis=0
            ) for attr in attributes
        }
    except ValueError as e:
        from IPython import embed; embed()  
    return eeg, stimuli
