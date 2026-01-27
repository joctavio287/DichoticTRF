from sklearn.model_selection import KFold
from typing import Dict, List, Optional
from datetime import datetime
from itertools import product
from pathlib import Path
import numpy as np
import logging

from utils.helpers_processing import (
    load_json_to_dict, dump_dict_to_json,
    fill_missing_nested, create_default_dict
)
from utils.helpers_figures import hyperparameter_selection
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
logger_val = setup_logger(
    name='validation',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)

# Use it
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_val)
    
    # Update logger to reflect overridden LOG_LEVEL
    new_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger_val.setLevel(new_level)
    for handler in logger_val.handlers:
        handler.setLevel(new_level)

# Start execution
def main(
    val_correlation_limit_percentage: float = config.VALIDATION_LIMIT_PERCENTAGE,
    overwrite_attributes: bool = config.OVERWRITE_EXISTING_ATTRIBUTES,
    attribute_preprocess: str = config.ATTRIBUTE_PREPROCESS,
    attributes_params: Dict = config.ATTRIBUTE_PARAMS,
    overwrite_results: bool = config.OVERWRITE_RESULTS,
    overwrite_figures: bool = config.OVERWRITE_FIGURES,
    validation_dir: Path = config.VALIDATION_DIR,
    eeg_preprocess: str = config.EEG_PREPROCESS,
    figures_dir: Path = config.FIGURES_DIR,
    alphas_grid: List[float] = config.ALPHAS_GRID,
    relevant_indexes: Optional[np.ndarray] = None,
    delays: np.ndarray = config.DELAYS,
    attributes: List[str] = config.ATTRIBUTES,
    times: np.ndarray = config.TIMES,
    subjects: List[Path] = config.SUBJECTS,
    n_folds: int = config.N_FOLDS,
    solver: str = config.SOLVER,
    bands: List[str] = config.BAND_FREQ,
    logger_val: logging.Logger = logger_val,
    sides: List[str] = config.SIDES,
    load_results: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Main function to perform validation analysis over subjects, bands, attributes and sides.
    
    Parameters
    ----------
    val_correlation_limit_percentage : float
        The percentage limit for correlation to consider for alpha selection.
    overwrite_attributes : bool
        Whether to overwrite existing attributes.
    attribute_preprocess : str
        The preprocessing method for attributes.
    target_sampling_rate : int
        The target sampling rate for the stimuli.
    attributes_params : Dict
        Additional parameters for attribute extraction.
    overwrite_results : bool
        Whether to overwrite existing results.
    overwrite_figures : bool
        Whether to overwrite existing figures.
    validation_dir : Path
        The directory where validation results are stored.
    eeg_preprocess : str
        The preprocessing method for EEG data.
    figures_dir : Path
        The directory where figures are saved.
    alphas_grid : List[float]
        The grid of alpha values to test.
    relevant_indexes : np.ndarray, optional
        Relevant indexes for EEG data selection.
    delays : np.ndarray
        The delays to consider in the model.
    attributes : List[str]
        The list of attributes to validate.
    times : np.ndarray
        The time points corresponding to the delays.
    subjects : List[Path]
        The list of subject paths to process.
    n_folds : int
        The number of folds for cross-validation.
    solver : str
        The solver to use in the model.
    bands : List[str]
        The frequency bands to consider.
    logger_val : logging.Logger
        The logger for validation process.
    sides : List[str]
        The side names to process ('left', 'right', 'mono').

    Returns
    -------
        Dict[str, Dict[str, Dict[str, float]]]
            A nested dictionary with subjects as keys, containing bands, sides, and attributes with their selected alpha values.
    """
    # Load existing validation results if available
    validation_path = validation_dir / f'X_{attribute_preprocess}_Y_{eeg_preprocess}' / f'correlation_limit_{val_correlation_limit_percentage}.json'
    alphas = load_json_to_dict(filepath=validation_path) if validation_path.exists() else {}
    if load_results:
        log_stage(
            f"Loading existing validation results from {validation_path}", 
            logger=logger_val
        )
        # Fill missing entries with None
        alphas_missing = fill_missing_nested(
            data_name="Validation results (alphas)",
            data=alphas,
            levels=(bands, attributes, sides, subjects),
            level_names=('band','attribute','side','subject'),
            default_value=None,
            log_stage=log_stage,
            logger=logger_val,
            log_level="WARNING"
        )
        return alphas_missing
    
    # Fill missing entries with None
    alphas = fill_missing_nested(
        data_name="Validation results (alphas)",
        data=alphas,
        levels=(bands, attributes, sides, subjects),
        level_names=('band','attribute','side','subject'),
        default_value=None,
        log_stage=log_stage,
        logger=logger_val,
        log_level="WARNING"
    )
    total_results = alphas.copy()
    plot_data = create_default_dict(
        levels=(bands, attributes, sides),
        default_value=None
    )

    # Store runtimes
    stimulus_runtimes = {}
    start_time = datetime.now().replace(microsecond=0)    

    for band in bands:
        log_memory_usage(logger=logger_val)
        for side in sides:
            for attribute in attributes:
                log_stage(
                    f"Current model: {band}-{attribute}-{side.capitalize()}", logger=logger_val
                )
                for subj_idx, subject_id in enumerate(subjects):
                    # Start time for this attribute
                    start_time_subj = datetime.now().replace(microsecond=0)

                    # Select alpha for this subject, band, attribute and side
                    alpha = alphas[band][attribute][side][subject_id]
   
                    # Load subject's data
                    eeg, stimulus = listening_data(                        
                        participant_id=subject_id,
                        band_freq=band,
                        attributes=attributes,
                        attribute_params=attributes_params,
                        overwrite=overwrite_attributes,
                        side=side
                    )

                    # Check if we need to skip because results exist
                    if alpha is not None and not overwrite_results:
                        log_stage(
                            f"SKIP-> {subj_idx} already computed ({alpha})", logger=logger_val
                        )
                        continue
                    
                    # Create placeholders for results
                    correlations_per_fold_train = np.zeros(
                        (n_folds, len(alphas_grid))
                    )
                    correlations_per_fold = np.zeros(
                        (n_folds, len(alphas_grid))
                    )
                    trfs_per_fold = np.zeros(
                        (n_folds, len(alphas_grid), len(delays))
                    )

                    # Make the Kfold test
                    kf_test = KFold(n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    if relevant_indexes is not None:
                        relevant_eeg = eeg[relevant_indexes]
                    else:
                        relevant_eeg = eeg
                    
                    # Run folds 
                    try:
                        for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                            logger_val.debug(f'\n\t······  [{fold+1}/{n_folds}]\t-->\t Validation fold')
                            trfs_per_fold[fold], correlations_per_fold[fold], correlations_per_fold_train[fold]  = fold_model(
                                fold=fold,
                                alpha=alphas_grid,
                                stims=stimulus[attribute],
                                eeg=relevant_eeg,
                                relevant_indexes=relevant_indexes,
                                train_indexes=train_indexes,
                                test_indexes=test_indexes,  
                                logger=logger_val,
                                solver=solver,
                                validation=True,
                                attribute_preprocess=attribute_preprocess,
                                eeg_preprocess=eeg_preprocess,
                                delays=delays
                            )     
                    except Exception as e:
                        from IPython import embed; embed()
                    # Aggregate results
                    correlations = np.nan_to_num(
                        np.nanmean(correlations_per_fold, axis=0)
                    )
                    correlations_std = np.nan_to_num(
                        np.nanstd(correlations_per_fold, axis=0)
                    )
                    correlations_train = np.nan_to_num(
                        np.nanmean(correlations_per_fold_train, axis=0)
                    )
                    trfs = np.nanmean(
                        trfs_per_fold, axis=0
                    )

                    # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
                    relative_difference = abs(
                        (correlations.max() - correlations)/correlations.max()
                    )
                    good_indexes_range = np.where(
                        relative_difference < val_correlation_limit_percentage
                    )[0]

                    # Get the very last one, because the greater the alpha, the smoothest the signal gets
                    alpha_selected = alphas_grid[int(good_indexes_range[-1])]
                    
                    # Subject logging
                    log_progress(
                        subj_idx + 1, 
                        len(subjects), 
                        message=f"α selected: {alpha_selected}", 
                        logger=logger_val,
                        start_time=start_time
                    )

                    # Calculate runtime for this stimulus
                    subj_runtime = datetime.now().replace(microsecond=0) - start_time_subj.replace(microsecond=0)
                    stimulus_runtimes[f"{band}_{attribute}_{side}_subject_{subj_idx}"] = subj_runtime
                    log_stage(
                        f"Runtime={subj_runtime} | n_alphas={len(alphas_grid)}",
                        logger=logger_val
                    )

                    # Make the alpha selection process plot
                    figures_path = figures_dir / 'validation' / f'subject_{subject_id}' / band.lower() / attribute.lower() / side.lower()
                    figures_path.mkdir(parents=True, exist_ok=True)
                    plot_data[band][attribute][side] = dict(
                        alphas_grid=alphas_grid,
                        correlations=correlations, 
                        correlations_std=correlations_std,
                        correlations_train=correlations_train, 
                        trfs=trfs,
                        times=times,
                        alpha_subject=alpha_selected,
                        correlation_limit_percentage=val_correlation_limit_percentage, 
                        subject=subject_id, 
                        attribute=attribute+'_'+side, 
                        band=band, 
                        save_path=figures_path/'validation_hyperparameter_selection.png',
                        overwrite=overwrite_figures,
                    )
                    
                    # Update dictionaries
                    total_results[band][attribute][side][subject_id] = alpha_selected
                    alphas[band][attribute][side][subject_id] = alpha_selected

                    # Save/store results
                    if overwrite_results:
                        dump_dict_to_json(filepath=validation_path, data_dict=alphas)
                    
        log_memory_usage(logger=logger_val)

    # Get total run time            
    total_runtime = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)

    # Make plots
    if overwrite_results:
        for band, attribute, side in product(bands, attributes, sides):
            hyperparameter_selection(**plot_data[band][attribute][side])
        log_stage(
            f"Figures saved in {figures_dir}/validation", 
            logger=logger_val,
            level="WARNING"
        )
    else:
        log_stage(
            "Overwrite_results is False.\n In this case, plots can't be generated because correlations and trfs are not saved.", 
            logger=logger_val,
            level="WARNING"
        )

    # Build summary (includes left/right in keys)
    summary_lines = [f"Summarize of runtimes for {Path(__file__).stem}"]
    for key in sorted(stimulus_runtimes.keys()):
        summary_lines.append(f"- {key}: {stimulus_runtimes[key]}\n")
    summary_lines.append(f"Total runtime: {total_runtime}")
    text = "\n".join(summary_lines)

    # Send text to telegram bot
    tel_message(
        api_token=API_TOKEN,
        chat_id=CHAT_ID, 
        message=text,
        logger=logger_val,
        verbose=True if logger_val.level > 20 else False
    )

    # Dump metadata
    log_path = get_logger_file_paths(logger_val)[0]
    dump_dict_to_json(
        filepath=str(log_path).replace('.log', '_metadata.json'),
        data_dict=config,
        module=True
    )

    # Print the completion message (single log)
    log_stage(text, logger=logger_val)
    return total_results

if __name__=='__main__':
    results = main()