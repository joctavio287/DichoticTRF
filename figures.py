import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import mne

from main import main as main_function

from utils.helpers_processing import (
    get_info_mne, clustering_by_correlation
)
from utils.logs import setup_logger, log_stage
from utils.helpers_figures import (
    evoked_potential_plot, heatmap_topoplot,
    ALLOWED_CLUSTERING_CORRELATION, trf_heatmap_plot
)
import config

logger_figures = setup_logger(
    name='figures',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR,
    level=config.LOG_LEVEL
)

def main(
    bands: list = config.BAND_FREQ,
    attributes: list = config.ATTRIBUTES,
    sides: list = config.SIDES,
    subjects: list = config.SUBJECTS,
    fig_dir: str = config.FIGURES_DIR,
    channel_selection: str = config.CHANNEL_SELECTION,
    times: np.ndarray = config.TIMES,
    logger_figures=logger_figures,
    extension: str = "png",
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
    channel_index, info_mne = get_info_mne(
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
                trfs = results[band][attribute][side]['trfs'] 
                if any(x is None for x in (correlations_std, correlations, metadata, trfs)):
                    log_stage(
                        f"Missing data for {band}-{attribute}-{side}, skipping.", 
                        logger=logger_figures,
                        level="WARNING"
                    )
                    continue
                ddof = metadata["number_of_channels"]-1
                
                # Generate average figures across subjects
                fig_dir_subfolder = fig_dir / "main_analysis" / band.lower() / attribute.lower() / side.lower()
                fig_dir_subfolder.mkdir(parents=True, exist_ok=True)

                # =============================================
                # Average TRF plot across subjects and features
                averaged_evoked_trf = mne.EvokedArray(
                    data=trfs.mean(axis=(0, 2)),
                    info=info_mne
                )
                if not evoked_potential_plot(
                        output_filepath=fig_dir_subfolder / f"average_trf_across_features.{extension}",
                        evoked=averaged_evoked_trf,
                        time_window=times
                    ):
                    log_stage(
                        f"Failed to create average TRF plot for {band}-{attribute}-{side}.",
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
                    trf_heatmap_plot(
                        output_filepath=fig_dir_subfolder / f"average_trf_across_channels.{extension}",
                        data=trfs_mean_across_channels,
                        time_window=times,
                        attribute=attribute,
                        order=order if attribute in ALLOWED_CLUSTERING_CORRELATION else None
                    )

                # ============================================
                # Average correlation topoplot across subjects
                average_correlation = correlations.mean(axis=0)
                if not heatmap_topoplot(
                        coefficient_values=average_correlation,
                        coefficient_name='Correlation',
                        info=info_mne,
                        figsize=(6,5),
                        figkwargs={},
                        colors='OrRd',
                        show=False,
                        output_filepath=fig_dir_subfolder / f"average_correlation_topomap.{extension}",
                        dpi=dpi
                    ):
                    log_stage(
                        f"Failed to create average correlation topoplot for {band}-{attribute}-{side}.",
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

                if not heatmap_topoplot(
                        coefficient_values=similarity,
                        coefficient_name='Similarity',
                        info=info_mne,
                        figsize=(6,5),
                        figkwargs={},
                        colors="Greens",
                        show=False,
                        output_filepath=fig_dir_subfolder / f"similarity_topomap.{extension}",
                        dpi=dpi
                    ):
                    log_stage(
                        f"Failed to create similarity topoplot for {band}-{attribute}-{side}.",
                        logger=logger_figures,
                        level="ERROR"
                    )

                log_stage(
                    f"Figures saved in {fig_dir_subfolder}.", logger=logger_figures, level="INFO"
                )

                # -> TRF topomaps en relevant times
                # -> matriz de similaridad

                # tfce?

                # individual_plot()
                # - > topoplot correlacion
                # - > TRFs

if __name__== "__main__":
    main()

        