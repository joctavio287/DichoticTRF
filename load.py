from pathlib import Path
import numpy as np
import logging
import mne

from utils.helpers_processing import (
    load_json_to_dict, fir_filter,
    custom_resample, band_selection
)
from utils.helpers_audio import (
    read_ogg, compute_single_stimulus
)
import config

# Notification bot
from utils.notification_telegram import tel_message 
from utils.telegram_config import API_TOKEN, CHAT_ID

# Command line and logging
from utils.from_commands import create_dynamic_parser, apply_args_to_config
from utils.logs import (
    log_stage, log_progress, log_memory_usage,
    get_logger_file_paths, setup_logger
)

# Initialize logger
logger_load= setup_logger(
    name='load',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)

def main(
    stimuli_dir: Path = config.STIMULI_DIR,
    attributes: list = config.ATTRIBUTES,
    attribute_params: dict = config.ATTRIBUTE_PARAMS,
    target_sample_rate: int = config.TARGET_SAMPLING_RATE,
    preprocessed_listening_dir: Path = config.PREPROCESSED_LISTENING_DIR,
    preprocessed_eeg_dir: Path = config.PREPROCESSED_DIR,
    bands: list = config.BAND_FREQ,
    logger_load: logging.Logger = logger_load
):

    # =================================
    # COMPUTE ATTRIBUTES FOR ALL AUDIOS
    audio_files = list(
        (stimuli_dir / "processed_audios" / "with_probe").glob("*.ogg")
    )
    # Compute attributes for all audios
    for attribute, attribute_params in zip(attributes, attribute_params.values()):
        for n_file, audio_file in enumerate(audio_files):
            log_progress(n_file + 1, len(audio_files), message=f'Audio Progress for {attribute}', logger=logger_load)
            # Load audio file
            sample_rate, audio = read_ogg(
                file_path=audio_file, 
                return_sample_rate=True
            )  
            for track, side in enumerate(['left', 'right', 'mono']):
                # Select audio side
                audio_side = audio[:, track] if side != 'mono' else np.mean(audio, axis=1)
                
                # Compute attribute values
                attribute_values = compute_single_stimulus(
                    axis=0,
                    sidename=side,
                    audio=audio_side,
                    attribute=attribute,
                    params=attribute_params,
                    sample_rate=sample_rate,
                    audio_filepath=audio_file,
                    target_sample_rate=target_sample_rate,
                )

                # Save attribute values and metadata
                save_path = preprocessed_listening_dir / attribute.lower() / side / f"{audio_file.stem[:7]}_{side}.npz"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    save_path,
                    attribute_values=attribute_values,
                    sample_rate=target_sample_rate,
                    attribute=attribute,
                    params=attribute_params,
                    audio_filepath=str(audio_file),
                    side=side
                )

    # ====================================
    # PREPROCESS EEG DATA FOR EACH SUBJECT 
    eeg_save_dir = preprocessed_listening_dir / "eeg" 
    eeg_filelist = list(
        (preprocessed_eeg_dir / 'fif').glob("*_preprocessed.fif")
    )

    # Process each subject's EEG data
    for n_file, eeg_file in enumerate(eeg_filelist):
        eeg_name = eeg_file.stem.replace(
            "_preprocessed", ""
        )
        log_progress(n_file + 1, len(eeg_filelist), message='EEG Progress', logger=logger_load)

        # Read preprocessed EEG data
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose='CRITICAL')
        raw = raw.pick_types(eeg=True, exclude=['M1', 'M2'])
        info_mne = raw.info.copy()

        # Read json annotation file
        annotations_data = load_json_to_dict(
            filepath=preprocessed_eeg_dir / 'annotations' / f"events_{eeg_name}.json"
        )
        # Set annotations to raw data
        raw = raw.set_annotations(
            mne.Annotations(
                onset=annotations_data['listening_onsets'],
                duration=annotations_data['listening_durations'],
                description=annotations_data["listening_annotations"]
            )
        )
        # Crop raw data by annotations
        raw_segments = raw.crop_by_annotations()

        # Synchronize and preprocess each segment
        for n_segment, raw_segment in enumerate(raw_segments):
            for band in bands:
                # Filter raw eeg data
                if band == 'FullBand':
                    delay, filtered_data = 0, raw_segment.get_data().T
                else:
                    l_freq, h_freq = band_selection(band=band)
                    delay, filtered_data = fir_filter(
                        array=raw_segment.get_data().T,
                        sfreq=int(raw_segment.info['sfreq']),
                        l_freq=l_freq,
                        h_freq=h_freq,
                        axis=0,
                        call_type="forward_compensated_reflected",
                        return_delay=True
                    )
                # Resample EEG to target sampling rate
                filtered_data = custom_resample(
                    array=filtered_data,
                    original_sr=int(raw_segment.info['sfreq']),
                    target_sr=target_sample_rate,
                    axis=0
                )
                
                # Save processed EEG data
                save_eeg_path = eeg_save_dir / band.lower() / f"{eeg_name}_segment{n_segment+1}_eeg_{band.lower()}.npz"
                save_eeg_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    save_eeg_path,
                    eeg=filtered_data,
                    sfreq=target_sample_rate,
                    delay_to_compensate=delay,
                    info=info_mne,
                    band=f"{band}: {l_freq}-{h_freq} Hz",
                    resampled=f"from {int(raw_segment.info['sfreq'])} Hz to {target_sample_rate} Hz"
                )

if __name__ == "__main__":
    main()