from datetime import datetime
import concurrent.futures
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
    log_stage, log_progress,
    setup_logger
)

# Initialize logger
logger_load= setup_logger(
    name='load',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)

# Use it
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_load)
    
    # Update logger to reflect overridden LOG_LEVEL
    new_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger_load.setLevel(new_level)
    for handler in logger_load.handlers:
        handler.setLevel(new_level)

def main(
    stimuli_dir: Path = config.STIMULI_DIR,
    attributes: list = config.ATTRIBUTES,
    attribute_params: dict = config.ATTRIBUTE_PARAMS,
    target_sample_rate: int = config.TARGET_SAMPLING_RATE,
    preprocessed_listening_dir: Path = config.PREPROCESSED_LISTENING_DIR,
    preprocessed_eeg_dir: Path = config.PREPROCESSED_EEG_DIR,
    bands: list = config.BAND_FREQ,
    over_write: bool = config.OVERWRITE_EXISTING_ATTRIBUTES,
    logger_load: logging.Logger = logger_load,
    parallel_workers: int = config.PARALLEL_WORKERS_LOADING
):
    """
    Load and preprocess data: compute attributes for audio files and preprocess EEG data.
    Parameters
    ----------
        stimuli_dir: Path
            Directory containing stimuli audio files.
        attributes: list
            List of attributes to compute for audio files.
        attribute_params: dict
            Dictionary of parameters for each attribute.
        target_sample_rate: int
            Target sampling rate for audio and EEG data.
        preprocessed_listening_dir: Path
            Directory to save preprocessed listening data.
        preprocessed_eeg_dir: Path
            Directory containing preprocessed EEG files.
        bands: list
            List of frequency bands for EEG preprocessing.
        over_write: bool
            If True, existing preprocessed files will be overwritten.
        logger_load: logging.Logger
            Logger instance for logging messages.
        parallel_workers: int
            Number of parallel workers for processing. If 0, no parallelism is used.
    Returns
    -------
        None
    """
    if not over_write: 
        log_stage(
            "Overwrite is set to False. Existing files will not be recomputed.", 
            logger=logger_load
        )

    # =================================
    # COMPUTE ATTRIBUTES FOR ALL AUDIOS
    audio_files = list(
        (stimuli_dir / "processed_audios" / "with_probe").glob("*.ogg")
    )
    # Precompute which attribute/side/audio combinations are needed
    needed = []
    for attribute, attribute_params in zip(attributes, attribute_params.values()):
        for audio_file in audio_files:
            for side in ['left', 'right', 'mono']:
                save_path = preprocessed_listening_dir / attribute.lower() / side / f"{audio_file.stem[:7]}_{side}.npz"
                if not (save_path.exists() and not over_write):
                    needed.append((attribute, attribute_params, audio_file, side, save_path))
                else:
                    log_stage(
                        f"Attribute {attribute} for {side} of {audio_file.stem[:7]} already computed. Skipping computation.", 
                        logger=logger_load, 
                        level='WARNING'
                    )

    # Compute attributes for all audios
    start_time = datetime.now().replace(microsecond=0)
    total = len(needed)
    def process_attribute(args):
        attribute, attribute_params, audio_file, side, save_path = args
        log_progress(
            needed.index(args) + 1, total, 
            message=f"Processing {attribute} {side} {audio_file.stem[:7]}", 
            start_time=start_time
        )
        sample_rate, audio = read_ogg(
            file_path=audio_file, 
            return_sample_rate=True
        )
        track = {'left': 0, 'right': 1, 'mono': None}[side]
        audio_side = audio[:, track] if side != 'mono' else np.mean(audio, axis=1)
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

    if parallel_workers > 0 or parallel_workers == -1:
        if parallel_workers == -1:
            import os
            parallel_workers = max_workers = int(0.8 * (os.cpu_count() - 1))
        log_stage(
            f"Processing attributes in parallel using {parallel_workers} workers.",
            logger=logger_load,
            level="INFO"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            list(executor.map(process_attribute, needed))
    else:
        log_stage(
            "Processing attributes sequentially without parallelism.",
            logger=logger_load,
            level="INFO"
        )
        for args in needed:
            process_attribute(args)
    # ====================================
    # PREPROCESS EEG DATA FOR EACH SUBJECT 
    eeg_save_dir = preprocessed_listening_dir / "eeg" 
    eeg_filelist = list(
        (preprocessed_eeg_dir / 'fif').glob("*_preprocessed.fif")
    )

    # Precompute which EEG/segment/band combinations are needed
    needed_eeg = []
    for eeg_file in eeg_filelist:
        eeg_name = eeg_file.stem.replace("_preprocessed", "")
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose='CRITICAL')
        raw = raw.pick_types(eeg=True, exclude=['M1', 'M2'])
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
        for n_segment, raw_segment in enumerate(raw_segments):
            for band in bands:
                save_eeg_path = eeg_save_dir / band.lower() / f"{eeg_name}_segment{n_segment+1}_eeg_{band.lower()}.npz"
                if not (save_eeg_path.exists() and not over_write):
                    needed_eeg.append((eeg_name, raw_segment, band, save_eeg_path, raw_segment.info.copy()))
                else:
                    log_stage(
                        f"EEG data for {eeg_name}-segment{n_segment+1}-{band} already computed. Skipping computation.", 
                        logger=logger_load, 
                        level='WARNING'
                    )

    # Process each subject's EEG data
    start_time = datetime.now().replace(microsecond=0)
    for n_file, (eeg_name, raw_segment, band, save_eeg_path, info_mne) in enumerate(needed_eeg):
        log_progress(
            n_file + 1, 
            len(needed_eeg), 
            message='EEG Progress', 
            logger=logger_load, 
            start_time=start_time
        )
        if band == 'FullBand':
            delay, filtered_data = 0, raw_segment.get_data().T
            l_freq, h_freq = None, None
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
    tel_message(
        api_token=API_TOKEN,
        chat_id=CHAT_ID,
        message=f"Loading and preprocessing completed.\nTotal audio attributes computed: {total}\nTotal EEG segments preprocessed: {len(needed_eeg)}"
    )

if __name__ == "__main__":
    main()