from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import mne

from utils.helpers_processing import (
    load_json_to_dict, fir_filter,
    custom_resample, band_selection
)
from utils.helpers_audio import read_ogg
import config

TRIGGER_DIR = config.PREPROCESSED_DIR / 'triggers'
ANNOT_DIR = config.PREPROCESSED_DIR / 'annotations'
FIF_DIR = config.PREPROCESSED_DIR / 'fif'


eeg_filelist = list(FIF_DIR.glob("*_preprocessed.fif"))
for eeg_file in tqdm(eeg_filelist, desc="Preprocessing listening data", total=len(eeg_filelist)):
    eeg_name = eeg_file.stem.replace(
         "_preprocessed", ""
    )
    # Load behavioural data 
    behavioural_data = load_json_to_dict( #FIXME: usar la misma convención de nombres para psychopy y biosemi
        config.BEHAVIOURAL_DIR / f"{eeg_name.split('prueba')[1]}_behavioural.json"
    )
    # Read preprocessed EEG data
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=config.VERBOSE_LEVEL)
    raw = raw.pick_types(eeg=True, exclude=['M1', 'M2'])
    info_mne = raw.info.copy()

    # Read json annotation file
    annotations_data = load_json_to_dict(
        filepath=ANNOT_DIR / f"events_{eeg_name}.json"
    )
    raw = raw.set_annotations(
        mne.Annotations(
            onset=annotations_data['listening_onsets'],
            duration=annotations_data['listening_durations'],
            description=annotations_data["listening_annotations"]
        )
    )
    raw_segments = raw.crop_by_annotations()
    eeg_save_dir = config.PREPROCESSED_LISTENING_DIR/ f"{eeg_name}" 
    audio_save_dir = config.PREPROCESSED_LISTENING_DIR/ f"{eeg_name}" / "audios"
    audio_save_dir.mkdir(parents=True, exist_ok=True)
    for n_segment, raw_segment in enumerate(raw_segments):
        # Load stimuli and eeg
        audio_sr, audio = read_ogg(
            file_path=behavioural_data['metadata']['audio_filepath'][n_segment],
            return_sample_rate=True
        )
        # Save processed audio data
        np.savez(
            audio_save_dir / f"{eeg_name}_segment{n_segment+1}_audios.npz",
            audio_l=audio[:, 0],
            audio_r=audio[:, 1],
            sfreq=audio_sr
        )
        # Resample audio EEG to target sampling rate
        raw_segment_data = custom_resample(
            array=raw_segment.get_data().T,
            original_sr=int(raw_segment.info['sfreq']),
            target_sr=config.TARGET_SAMPLING_RATE,
            axis=0
        )
        for band in config.BAND_FREQ:
            l_freq, h_freq = band_selection(band=band)
            if band == 'FullBand':
                delay, filtered_data = 0, raw_segment_data
            else:
                # Filter and resample raw eeg data
                delay, filtered_data = fir_filter(
                    array=raw_segment_data,
                    sfreq=raw_segment.info['sfreq'],
                    l_freq=l_freq,
                    h_freq=h_freq,
                    axis=0,
                    call_type="forward_compensated_cut",
                    return_delay_cut=True
                )
            save_eeg_path = eeg_save_dir / band.lower() / f"{eeg_name}_segment{n_segment+1}_eeg_{band.lower()}.npz"
            save_eeg_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_eeg_path,
                eeg=filtered_data,
                sfreq=config.TARGET_SAMPLING_RATE,
                delay_to_compensate=delay,
                info=info_mne
            )