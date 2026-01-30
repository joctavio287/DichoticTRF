from sklearn.model_selection import KFold
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

from utils.helpers_processing import (
    load_json_to_dict, dump_dict_to_json,
    fill_missing_nested, create_default_dict
)
from utils.helpers_load import listening_data
from mtrf_models import fold_model
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
logger_main = setup_logger(
    name='main',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)

# Use it
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_main)
    
    # Update logger to reflect overridden LOG_LEVEL
    new_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger_main.setLevel(new_level)
    for handler in logger_main.handlers:
        handler.setLevel(new_level)

# Start execution
def main(
    val_correlation_limit_percentage: float = config.VALIDATION_LIMIT_PERCENTAGE,
    overwrite_attributes: bool = config.OVERWRITE_EXISTING_ATTRIBUTES,
    overwrite_results: bool = config.OVERWRITE_RESULTS,
    attribute_preprocess: str = config.ATTRIBUTE_PREPROCESS,
    number_of_channels: int = config.NUMBER_OF_CHANNELS,
    validation_dir: Path = config.VALIDATION_DIR,
    correlations_dir: Path = config.CORRELATIONS_DIR,
    attributes_params: Dict = config.ATTRIBUTE_PARAMS,
    eeg_preprocess: str = config.EEG_PREPROCESS,
    default_alpha: float = config.DEFAULT_ALPHA,
    attributes: List[str] = config.ATTRIBUTES,
    subjects: List[Path] = config.SUBJECTS,
    set_alpha: bool = config.SET_ALPHA,
    relevant_indexes: Optional[np.ndarray] = None,
    trfs_dir: Path = config.TRFS_DIR,
    bands: List[str] = config.BAND_FREQ,
    delays: np.ndarray = config.DELAYS,
    logger_main: logging.Logger = logger_main,
    sides: List[str] = config.SIDES,
    n_folds: int = config.N_FOLDS,
    solver: str = config.SOLVER,
    load_results: bool = False,
    number_of_segments: Optional[int] = config.MAX_NUMBER_OF_SEGMENTS,
    same_val: bool = config.SAME_VALIDATION
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Main function to perform TRFs analysis over subjects, bands, attributes and sides.

    Parameters
    ----------
    val_correlation_limit_percentage : float
        The percentage limit for correlation to consider for alpha selection.
    overwrite_attributes : bool
        Whether to overwrite existing attributes.
    target_sampling_rate : int
        The target sampling rate for the stimuli.
    attribute_preprocess : str
        The preprocessing method for attributes.
    number_of_channels : int
        The number of EEG channels.
    attributes_params : Dict
        Additional parameters for attribute extraction.
    overwrite_results : bool
        Whether to overwrite existing results.
    correlations_dir : Path
        The directory where correlation results are stored.
    relevant_indexes : np.ndarray, optional
        Relevant indexes for EEG data selection.
    validation_dir : Path
        The directory where validation results are stored.
    eeg_preprocess : str
        The preprocessing method for EEG data.
    default_alpha : float
        Default alpha value for the model.
    logger_main : logging.Logger
        The logger for main process.
    attributes : List[str]
        The list of attributes to validate.
    set_alpha : bool
        Whether to set all alphas to the default value.
    subjects : List[Path]
        The list of subject paths to process.
    sides : List[str]
        The side names to process ('left', 'right', 'mono').
    trfs_dir : Path
        The directory where TRF results are stored.
    n_folds : int
        The number of folds for cross-validation.
    delays : np.ndarray
        The delays to consider in the model.
    solver : str
        The solver to use in the model.
    bands : List[str]
        The frequency bands to consider.
    load_results : bool
        Whether to load existing results from disk.
    number_of_segments : Optional[int]
        The number of segments to use per subject. If None, use all segments.
    same_val : bool
        Whether to use the same regularization for every subject.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]
        Nested results keyed by band -> attribute -> side, containing arrays for
        'correlations', 'correlations_std', 'trfs', and a 'metadata' dict.
    """
    # Load existing validation results if available
    validation_path = validation_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}' / f'correlation_limit_{val_correlation_limit_percentage}.json'
    if set_alpha:
        alphas = create_default_dict(
            levels=(bands, attributes, sides, subjects),
            default_value=default_alpha
        )
        log_stage(f"Setting all alphas to default value: {default_alpha}", logger=logger_main)
    else:
        if validation_path.exists():
            alphas = load_json_to_dict(filepath=validation_path) 
            alphas = fill_missing_nested(
                data_name="Validation results (alphas)",
                data=alphas,
                levels=(bands, attributes, sides, subjects),
                level_names=('band','attribute','side','subject'),
                default_value=default_alpha,
                log_stage=log_stage,
                logger=logger_main,
                log_level="WARNING"
            )
        else:
            log_stage(f"No validation file found at {validation_path}. Using default alpha ({default_alpha}) for all subjects, bands, attributes, and sides.", logger=logger_main, level="WARNING")
            alphas = create_default_dict(
                levels = (bands, attributes, sides, subjects),
                default_value=default_alpha
            )
    total_results = create_default_dict(
        levels = (bands, attributes, sides),
        default_value=None
    )
    if load_results:
        log_stage("Loading existing results from disk...", logger=logger_main)
        for band in bands:
            for attribute in attributes:
                for side in sides:
                    correlations_path = correlations_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}' / f'correlations_{band.lower()}_{attribute.lower()}_{side}.npz'
                    trfs_path = trfs_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}' / f'trfs_{band.lower()}_{attribute.lower()}_{side}.npz'
                    if correlations_path.exists() and trfs_path.exists():
                        corr_data = np.load(correlations_path, allow_pickle=True)
                        trf_data =  np.load(trfs_path, allow_pickle=True)
                        from IPython import embed; embed()
                        trf_data['alphas']
                        total_results[band][attribute][side] = {
                            'correlations': corr_data['correlations'],
                            'correlations_std': corr_data['correlations_std'],
                            'alphas': trf_data['alphas'],
                            'trfs': trf_data['trfs'],
                            'metadata': {
                                **corr_data['metadata'].item(),**trf_data['metadata'].item()
                            }
                        }
                        log_stage(f"Loaded results for {band}-{attribute}-{side} from disk.", logger=logger_main)
                    else:
                        # Fill missing entries with None
                        log_stage(f"Results for {band}-{attribute}-{side} not found on disk. They will be filled with Nones.", logger=logger_main, level="WARNING")
                        total_results[band][attribute][side] = {
                            'correlations_std': None,
                            'correlations': None,
                            'metadata': None,
                            'alphas': None,
                            'trfs': None
                        }
        return total_results

    # Store runtimes
    stimulus_runtimes = {}
    start_time = datetime.now().replace(microsecond=0)    

    for band in bands:
        log_memory_usage(logger=logger_main)
        for side in sides:
            for attribute in attributes:
                log_stage(
                    f"Current model: {band}-{attribute}-{side.capitalize()}", logger=logger_main
                )
                # Start time for this attribute
                start_time_attribute = datetime.now().replace(microsecond=0)
                correlations_std = []
                correlations = []
                trfs = []
                for subj_idx, subject_id in enumerate(subjects):
                    # Select alpha for this subject, band, attribute and side
                    alpha = alphas[band][attribute][side][subject_id]

                    # Load subject's data
                    eeg, stimulus = listening_data(                        
                        participant_id=subject_id,
                        band_freq=band,
                        attributes=attributes,
                        attribute_params=attributes_params,
                        overwrite=overwrite_attributes,
                        side=side,
                        number_of_segments=number_of_segments
                    )
                    number_of_features = stimulus[attribute].shape[1]

                    # Create placeholders for results
                    correlations_per_fold = np.zeros(
                        (n_folds, number_of_channels)
                    )
                    trfs_per_fold = np.zeros(
                        (n_folds, number_of_channels, number_of_features, len(delays))
                    )

                    # Make the Kfold test
                    kf_test = KFold(n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    if relevant_indexes is not None:
                        relevant_eeg = eeg[relevant_indexes]
                    else:
                        relevant_eeg = eeg
                    
                    # Run folds 
                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        logger_main.debug(f'\n\t······  [{fold+1}/{n_folds}]\t-->\t Main fold')
                        trfs_per_fold[fold], correlations_per_fold[fold] = fold_model(
                            fold=fold,
                            alpha=alpha,
                            stims=stimulus[attribute],
                            eeg=relevant_eeg,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,  
                            logger=logger_main,
                            solver=solver,
                            attribute_preprocess=attribute_preprocess,
                            eeg_preprocess=eeg_preprocess,
                            delays=delays
                        )     
                    # Aggregate results
                    correlations.append(
                        np.nan_to_num(
                            np.nanmean(correlations_per_fold, axis=0)
                        )
                    )
                    correlations_std.append(
                        np.nan_to_num(
                            np.nanstd(correlations_per_fold, axis=0)
                        )
                    )
                    trfs.append(
                        np.nanmean(
                            trfs_per_fold, axis=0
                        )
                    )
                    # Subject logging
                    log_progress(
                        subj_idx + 1, 
                        len(subjects), 
                        message=f"α: {alpha}", 
                        logger=logger_main,
                        start_time=start_time_attribute
                    )
                # Calculate runtime for this stimulus
                att_runtime = datetime.now().replace(microsecond=0) - start_time_attribute.replace(microsecond=0)
                stimulus_runtimes[f"{band}_{attribute}_{side}"] = att_runtime

                correlations_std = np.stack(correlations_std, axis=0)
                correlations = np.stack(correlations, axis=0)
                trfs =  np.stack(trfs, axis=0)

                # Save/store results
                correlations_path = correlations_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}'
                trfs_path = trfs_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}' 
                correlations_path.mkdir(parents=True, exist_ok=True)
                trfs_path.mkdir(parents=True, exist_ok=True)
                if overwrite_results:
                    np.savez(
                        file=correlations_path/f'correlations_{band.lower()}_{attribute.lower()}_{side}.npz',
                        correlations=correlations,
                        correlations_std=correlations_std,
                        metadata={
                            'number_of_channels': number_of_channels, 
                            'number_of_subjects': len(subjects)
                        }
                    )
                    np.savez(
                        file=trfs_path/f'trfs_{band.lower()}_{attribute.lower()}_{side}.npz',
                        trfs=trfs,
                        alphas=alphas[band][attribute][side],
                        metadata={
                            'number_of_channels': number_of_channels, 
                            'number_of_subjects': len(subjects), 
                            'number_of_features': number_of_features, 
                            'number_of_delays': len(delays)
                        }
                    )

                # Update total results
                total_results[band][attribute][side] = {
                    'correlations': correlations,
                    'correlations_std': correlations_std,
                    'trfs': trfs,
                    'alphas': alphas[band][attribute][side],
                    'metadata': {
                        'number_of_channels': number_of_channels, 
                        'number_of_subjects': len(subjects), 
                        'number_of_features': number_of_features, 
                        'number_of_delays': len(delays)
                    }
                }    
                
                log_stage(
                    f"Runtime={att_runtime} | subjects={len(subjects)}",
                    logger=logger_main
                )
        log_memory_usage(logger=logger_main)

    # Get total run time            
    total_runtime = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)

    # Build summary (includes left/right in keys)
    summary_lines = [f"Summarize of runtimes for {Path(__file__).stem}:"]
    for key in sorted(stimulus_runtimes.keys()):
        summary_lines.append(f"- {key}: {stimulus_runtimes[key]}\n")
    summary_lines.append(f"Total runtime: {total_runtime}")
    text = "\n".join(summary_lines)

    # Send text to telegram bot
    tel_message(
        api_token=API_TOKEN,
        chat_id=CHAT_ID, 
        message=text,
        logger=logger_main,
        verbose=True if logger_main.level > 20 else False
    )

    # Dump metadata
    log_path = get_logger_file_paths(logger_main)[0]
    dump_dict_to_json(
        filepath=str(log_path).replace('.log', '_metadata.json'),
        data_dict=config,
        module=True
    )

    # Print the completion message (single log)
    log_stage(text, logger=logger_main)
    return total_results

if __name__=='__main__':
    results = main()