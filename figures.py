import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import logging
import mne

from main import main as main_function

from utils.helpers_processing import (
    get_info_mne, clustering_by_correlation
)
from utils.from_commands import create_dynamic_parser, apply_args_to_config
from utils.logs import setup_logger, log_stage, log_if_false
from utils.helpers_figures import (
    ALLOWED_CLUSTERING_CORRELATION, trf_heatmap_plot,
    evoked_potential_plot, topoplot,
)
import config

logger_figures = setup_logger(
    name='figures',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)
if __name__ == "__main__":
    parser = create_dynamic_parser()
    args = parser.parse_args()
    apply_args_to_config(args, logger=logger_figures)
    
    # Update logger to reflect overridden LOG_LEVEL
    new_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger_figures.setLevel(new_level)
    for handler in logger_figures.handlers:
        handler.setLevel(new_level)

def main(
    bands: list = config.BAND_FREQ,
    attributes: list = config.ATTRIBUTES,
    sides: list = config.SIDES,
    subjects: list = config.SUBJECTS,
    fig_dir: str = config.FIGURES_DIR,
    channel_selection: str = config.CHANNEL_SELECTION,
    times: np.ndarray = config.TIMES,
    logger_figures: logging.Logger = logger_figures,
    extension: str = "png",
    same_val: bool = config.SAME_VALIDATION,
    dpi: int = 300
):
    """
    Generate figures based on the TRF analysis results.

    Parameters
    ----------
    sides : list
        List of sides to process (e.g., ['left', 'right', 'mono']).
    """
    # Load results from the main function
    results = main_function(
        attributes=attributes,
        sides=sides,
        bands=bands,
        load_results=True
    )
    channel_indexes, info_mne = get_info_mne(
        channel_selection=channel_selection,
        return_channels_index=True
    )
    for band in results:
        for attribute in results[band]:
            for side in results[band][attribute]:
                plt.close('all')
                correlations_std = results[band][attribute][side]['correlations_std']
                correlations = results[band][attribute][side]['correlations']
                metadata = results[band][attribute][side]['metadata']
                alphas = results[band][attribute][side]['alphas'].item()
                trfs = results[band][attribute][side]['trfs'] 
                if any(x is None for x in (correlations_std, correlations, metadata, trfs)):
                    log_stage(
                        f"Missing data for {band}-{attribute}-{side}, skipping.", 
                        logger=logger_figures,
                        level="WARNING"
                    )
                    continue

                # Select only the desired channels
                correlations_std = correlations_std[:, channel_indexes]
                correlations = correlations[:, channel_indexes]
                trfs = trfs[:, channel_indexes, :, :]

                # Generate average figures across subjects
                fig_dir_subfolder = fig_dir / "main_analysis" / same_val / band.lower() / attribute.lower() / side.lower()
                fig_dir_subfolder.mkdir(parents=True, exist_ok=True)

                # =============================================
                # Average TRF plot across subjects and features
                averaged_evoked_trf = mne.EvokedArray(
                    data=trfs.mean(axis=(0, 2)),
                    info=info_mne
                )
                log_if_false(
                    evoked_potential_plot(
                        output_filepath=fig_dir_subfolder / f"average_trf_across_features.{extension}",
                        evoked=averaged_evoked_trf,
                        time_window=times
                    ),
                    f"{band}-{attribute}-{side}: TRF Failed",
                    logger=logger_figures,
                    level="ERROR"
                )
                # =============================================
                # Average TRF plot across subjects and channels
                if metadata['number_of_features'] > 1:
                    trfs_mean_across_channels = trfs.mean(axis=(0,1))
                    if attribute in ALLOWED_CLUSTERING_CORRELATION:
                        order = clustering_by_correlation(
                            data=trfs_mean_across_channels
                            )
                    log_if_false(
                        trf_heatmap_plot(
                            output_filepath=fig_dir_subfolder / f"average_trf_across_channels.{extension}",
                            data=trfs_mean_across_channels,
                            time_window=times,
                            attribute=attribute,
                            order=order if attribute in ALLOWED_CLUSTERING_CORRELATION else None
                        ),
                        f"{band}-{attribute}-{side}: TRF Heatmap Failed",
                        logger=logger_figures,
                        level="ERROR"
                    )

                # ============================================
                # Average correlation topoplot across subjects
                average_correlation = correlations.mean(axis=0)
                log_if_false(
                    topoplot(
                        coefficient_values=average_correlation,
                        coefficient_name='Correlation',
                        info=info_mne,
                        figsize=(6,5),
                        figkwargs={},
                        colors='OrRd',
                        show=False,
                        output_filepath=fig_dir_subfolder / f"average_correlation_topomap.{extension}",
                        dpi=dpi,
                        logger=logger_figures
                    ),
                    f"{band}-{attribute}-{side}: Average Correlation Topoplot Failed",
                    logger=logger_figures,
                    level="ERROR"
                )
                # ===================================
                # Similarity topoplot across subjects
                average_trfs_across_features = trfs.mean(axis=2)
                correlation_matrices = np.zeros(
                    shape=(
                        metadata['number_of_channels'], 
                        metadata['number_of_subjects'], 
                        metadata['number_of_subjects']
                    )
                )

                # Calculate correlation betweem subjects
                for channel in range(metadata['number_of_channels']):
                    matrix = average_trfs_across_features[:,channel,:] 
                    correlation_matrices[channel] = np.corrcoef(matrix)

                # Correlacion por canal
                similarity = np.zeros(metadata['number_of_channels'])
                for channel in range(metadata['number_of_channels']):
                    channel_corr_values = correlation_matrices[channel][
                        np.tril_indices(metadata['number_of_subjects'], k=-1)
                    ]
                    if channel_corr_values.size == 0 or np.all(np.isnan(channel_corr_values)):
                        similarity[channel] = np.nan
                    else:
                        similarity[channel] = np.nanmean(np.abs(channel_corr_values))

                log_if_false(
                    topoplot(
                        coefficient_values=similarity,
                        coefficient_name='Similarity',
                        info=info_mne,
                        figsize=(6,5),
                        figkwargs={},
                        colors="Greens",
                        show=False,
                        output_filepath=fig_dir_subfolder / f"similarity_topomap.{extension}",
                        dpi=dpi,
                        logger=logger_figures
                    ),
                    f"{band}-{attribute}-{side}: Similarity Topoplot Failed",
                    logger=logger_figures,
                    level="ERROR"
                )

                log_stage(
                    f"Figures saved in {fig_dir_subfolder}.", logger=logger_figures, level="INFO"
                )
                
                # ========================
                # Individual subject plots
                for subj_i, subject in enumerate(subjects):
                    fig_dir_subfolder_subject = fig_dir_subfolder / 'individual_plots' / str(subj_i)
                    fig_dir_subfolder_subject.mkdir(parents=True, exist_ok=True)
                    
                    # ================================
                    # Average TRF plot across features
                    averaged_evoked_trf = mne.EvokedArray(
                        data=trfs[subj_i, :, :, :].mean(axis=1),
                        info=info_mne
                    )
                    log_if_false(
                        evoked_potential_plot(
                            output_filepath=fig_dir_subfolder_subject / f"average_trf_across_features.{extension}",
                            evoked=averaged_evoked_trf,
                            time_window=times,
                            title=f"Subject {subject} - α: {alphas[subject]:.4f}"
                        ),
                        f"{band}-{attribute}-{side}-{subj_i}: TRF Failed",
                        logger=logger_figures,
                        level="ERROR"
                    )
                    # ================================
                    # Average TRF plot across channels
                    if metadata['number_of_features'] > 1:
                        trfs_mean_across_channels = trfs[subj_i, :, :, :].mean(axis=0)
                        if attribute in ALLOWED_CLUSTERING_CORRELATION:
                            order = clustering_by_correlation(
                                data=trfs_mean_across_channels
                                )
                        log_if_false(
                            trf_heatmap_plot(
                                output_filepath=fig_dir_subfolder_subject / f"average_trf_across_channels.{extension}",
                                data=trfs_mean_across_channels,
                                time_window=times,
                                attribute=attribute,
                                order=order if attribute in ALLOWED_CLUSTERING_CORRELATION else None,
                                title=f"Subject {subject} - α: {alphas[subject]:.4f}"
                            ),
                            f"{band}-{attribute}-{side}-{subj_i}: TRF Heatmap Failed",
                            logger=logger_figures,
                            level="ERROR"
                        )
                    # ====================
                    # Correlation topoplot
                    log_if_false(
                        topoplot(
                            coefficient_values=correlations[subj_i],
                            coefficient_name='Correlation',
                            info=info_mne,
                            figsize=(6,5),
                            figkwargs={},
                            colors='OrRd',
                            show=False,
                            output_filepath=fig_dir_subfolder_subject / f"average_correlation_topomap.{extension}",
                            dpi=dpi,
                            logger=logger_figures
                        ),
                        f"{band}-{attribute}-{side}-{subj_i}: Average Correlation Topoplot Failed",
                        logger=logger_figures,
                        level="ERROR"
                    )
                    


                # tfce?

                # individual_plot()
                # - > topoplot correlacion
                # - > TRFs

if __name__== "__main__":
    main()

        