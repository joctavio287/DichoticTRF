"""
Helpers for plotting figures
"""
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import cm
from pathlib import Path
from typing import Union
import numpy as np
import functools
import logging
import librosa
import mne

from utils.helpers_processing import get_gfp_peaks
from utils.logs import log_stage
from utils.helpers_audio import LEXICON
import config

if config.USE_SCIENCE_PLOTS:
    import scienceplots
    plt.style.use("science")

ALLOWED_CLUSTERING_CORRELATION = ['Phonemes']

def safe_plot(func):
    """
    Decorator to return False if the function raises any exception.
    
    Parameters
    ----------
    func : function
        The plotting function to be decorated.
    
    Returns
    -------
    bool
        True if the function executes successfully, False otherwise.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return False
    return wrapper

def feature_names_mapping(
    attribute: str,
    number_of_features: int
) -> dict:
    """
    Returns a mapping of feature indices to human-readable names for a given attribute.
    
    Parameters
    ----------
    attribute : str
        The attribute name for which to get feature names.
    
    Returns
    -------
    dict
        A dictionary mapping feature indices to names.
    """
    axis_mapping = {
        'ylabel': None,
        'yticks': None,
        'yticklabels': None
    }
    if attribute == 'Spectrogram':
        bands_center = librosa.mel_frequencies(n_mels=number_of_features+2, fmin=0, fmax=16000/2)[1:-1]
        axis_mapping['full_labels'] = [int(bands_center[i]) for i in range(number_of_features)]
        axis_mapping['ylabel'] = 'Frecuency (Hz)'
        axis_mapping['yticklabels'] = [int(bands_center[i]) for i in np.arange(0, len(bands_center), 2)]
        axis_mapping['yticks'] = np.arange(0, number_of_features, 2)
    elif attribute == 'Phonemes':
        # Using config.PHONEME_LIST
        axis_mapping['full_labels'] = LEXICON['phonemes']
        axis_mapping['ylabel'] = 'Phonemes'
        axis_mapping['yticklabels'] = LEXICON['phonemes']
        axis_mapping['yticks'] = np.arange(0, number_of_features, 1)
    return axis_mapping

@safe_plot
def onsets_plot(
    events_times: list[np.ndarray],
    events_labels: list[str],
    events_colors: Union[list[str], None] = None,
    output_filepath: Union[Path, str, None] = None,
    figsize: tuple = (10, 5),
    dpi: int = 300,
    xlim: Union[tuple, None] = None,
    show: bool = False,
    figkwargs: dict = {},
    verbose: bool = True
)-> None:
    """
    Plot event onsets over time.

    Parameters
    ----------
    events_times : list[np.ndarray]
        List of arrays of time points, corresponding to event onsets.
    output_filepath : Union[Path, str, None]
        Filepath to save the output figure. If None, the figure is not saved.
    events_labels : list[str]
        List of labels for each event type.
    events_colors : Union[list[str], None]
        List of colors for each event type. If None, default colors are used.
    figsize : tuple
        Size of the figure.
    dpi : int
        Dots per inch for the saved figure.
    xlim : Union[tuple, None]
        x-axis limits as (min, max). If None, uses full range.
    show : bool
        Whether to display the figure. This overrides saving if True.
    figkwargs : dict
        Additional keyword arguments for the figure.

    Returns
    -------
    bool
        True if the figure was created successfully.
    """
    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=figsize,
        **figkwargs
    )
    if events_colors is None:
        events_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(events_times)]
    _ = ax.eventplot(
        events_times,
        colors=events_colors,
        lineoffsets=np.arange(len(events_times)),
        linelengths=0.8,
        linewidths=1.5,
        alpha=0.8
    )
    _ = ax.set_yticks(np.arange(len(events_labels)))
    _ = ax.set_yticklabels(events_labels)
    if xlim is not None:
        _ = ax.set_xlim(xlim)
    _ = ax.set_xlabel("Time (s)")
    _ = ax.set_title("Event Onsets Over Time")
    
    if show:
        plt.show(block=True)
    elif output_filepath is not None:
        _ = fig.savefig(
            output_filepath, dpi=dpi
        )
    plt.close(fig)

    if verbose:
        print('\n\t Figure saved to: ', output_filepath)
    return True

@safe_plot
def evoked_potential_plot(
    evoked: mne.Evoked,
    output_filepath: Union[Path, str, None]=None,
    time_window: np.ndarray = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    show: bool = False,
    figkwargs: dict = {}
)-> bool:
    """
    Plot evoked potential with mean ERP.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data to plot.
    output_filepath : Union[Path, str, None]
        Filepath to save the output figure. If None, the figure is not saved.
    time_window : np.ndarray
        Time window in seconds for x-axis ticks. If None, uses full range of evoked.times
    figsize : tuple
        Size of the figure.
    dpi : int
        Dots per inch for the saved figure.
    show : bool
        Whether to display the figure. This overrides saving if True.
    figkwargs : dict
        Additional keyword arguments for the figure.

    Returns
    -------
    bool
        True if the figure was created successfully.
    """
    
    fig, [ax_topomaps, ax_timeseries] = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        constrained_layout=True,
        **figkwargs
    )
    # Shift time to start at time_window[0] if provided
    _ = evoked.shift_time(
        tshift=time_window[0] if time_window is not None else evoked.times[0],
        relative=True
    )
    # Peak detection
    peak_times, peak_amplitudes = get_gfp_peaks(evoked, min_dist_ms=25, rel_height=0.1)
    
    # Evoked potential time series
    evoked_plot = evoked.plot(
        scalings={'eeg':1},
        zorder='std',
        time_unit='ms',
        show=False,
        spatial_colors=True,
        axes=ax_timeseries,
        # units='uV',
        gfp=True
    )    
    
    # Mean
    mean_plot = ax_timeseries.plot(
        evoked.times*1e3, #ms
        evoked._data.mean(axis=0),
        color='black',
        label='Mean',
        zorder=130,
        linewidth=1.2
    )
    # Standard Error of the Mean (SEM)
    error_line = evoked._data.std(axis=0, ddof=1)/np.sqrt(evoked._data.shape[0])
    stdofmean_plot = ax_timeseries.fill_between(
        evoked.times*1e3, #ms
        evoked._data.mean(axis=0) - error_line,
        evoked._data.mean(axis=0) + error_line,
        color='black',
        label='S.E.M',
        alpha=0.3,
        zorder=120,
        linewidth=1.2
    )
    # Add vertical lines at peak times
    for peak_time in peak_times:
        _ = ax_timeseries.axvline(
            x=peak_time*1e3,  # ms
            color='Black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            label='GFP peaks' if peak_time == peak_times[0] else ""
        )
    # Remove "Nave" tag
    for txt in fig.findobj(mtext.Text):
        if "ave" in txt.get_text():
                txt.remove()
    
    # Axes labels and limits
    maximum = np.max(np.abs(evoked._data)) 
    _ = ax_timeseries.set_ylim(-maximum, maximum)
    _ = ax_timeseries.set_ylabel('Amplitude (a.u.)')
    _ = ax_timeseries.set_xlabel('Time (ms)')

    time_window = time_window if time_window is not None else evoked.times
    start_ms = int(np.round(time_window[0] * 1e3 / 100) * 100)
    end_ms = int(np.round(time_window[-1] * 1e3 / 100) * 100)
    _ = ax_timeseries.set_xticks(
        np.arange(
            start_ms,
            end_ms + 100,
            100
        )
    )
    _ = ax_timeseries.legend(loc='upper right', fontsize=12)
    _ = ax_timeseries.grid(True)

    # Clear the top axis (we will use insets aligned to time)
    ax_topomaps.set_axis_off()

    # Normalize time to the range of the bottom axis
    tmin, tmax = time_window[0], time_window[-1]

    # Global vlim so all use the same color scale
    all_topo = evoked.copy().get_data()
    vmax = np.nanmax(np.abs(all_topo))
    vlim = (-vmax, vmax)
    
    first_im = None
    for peak_time, peak_amplitude in zip(peak_times, peak_amplitudes):
        # Normalized x position for inset
        x_norm = (peak_time - tmin) / (tmax - tmin)
        x_norm = np.clip(x_norm, 0.0, 1.0)

        # Create inset topomap
        inset = ax_topomaps.inset_axes(
            [x_norm - 0.08, 0.05, 0.16, 0.9] # [x0, y0, w, h] in fraction of parent axes
        )  
        im, _ = mne.viz.plot_topomap(
            data=evoked.copy().get_data()[:, evoked.time_as_index(peak_time)].reshape(-1),
            pos=evoked.info,
            axes=inset,
            show=False,
            sphere=0.07,
            cmap='RdBu_r',
            extrapolate='local',
            border=0,
            vlim=vlim
        )
        _ = inset.set_title(f'{peak_time*1e3:.0f} ms', fontsize=8)
        if first_im is None: first_im = im

    # first_im contains the first topomap image for colorbar
    if first_im is not None: 
        cax = fig.add_axes([0.1, 0.60, 0.3, 0.015]) # left, bottom, width, height
        cbar = fig.colorbar(first_im, cax=cax, orientation='horizontal')
        _ = cbar.set_label('Amplitude (a.u.)')
    if show:
        plt.show(block=True)
    elif output_filepath is not None:
        _ = fig.savefig(
            output_filepath,
            dpi=dpi
        )
    plt.close(fig)
    return True

@safe_plot
def trf_heatmap_plot(
    data: np.ndarray,
    attribute: str,
    time_window: np.ndarray,
    order: Union[np.ndarray, None] = None,
    output_filepath: Union[Path, str, None]=None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    show: bool = False,
    figkwargs: dict = {}
) -> bool:
    """
    Plot TRF heatmap with individual feature time series.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (number_of_features, number_of_timepoints) containing TRF data.
    attribute : str
        The attribute name for feature labeling.
    time_window : np.ndarray
        1D array of time points corresponding to the TRF data.
    order : Union[np.ndarray, None]
        Optional array specifying the order of features for clustering. If None, no reordering is applied.
    output_filepath : Union[Path, str, None]
        Filepath to save the output figure. If None, the figure is not saved.
    figsize : tuple
        Size of the figure.
    dpi : int
        Dots per inch for the saved figure.
    show : bool
        Whether to display the figure. This overrides saving if True.
    figkwargs : dict
        Additional keyword arguments for the figure.

    Returns
    -------
    bool
        True if the figure was created successfully.
    """
    
    number_of_features = data.shape[0]
    c = feature_names_mapping(
        attribute=attribute, number_of_features=number_of_features
    )
    if attribute in ALLOWED_CLUSTERING_CORRELATION and order is not None:
        # Reorder data based on clustering
        data_ordered = data[order, :]
        c['yticks'] = c['yticks'][order]
        c['yticklabels'] = [c['yticklabels'][i] for i in order]
    else:
        data_ordered = data
        order = np.arange(number_of_features)

    # Create figure and title
    fig, [ax_timeseries, ax_heatmap] = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        **figkwargs
    )
    # Time series plot
    colors = cm.get_cmap('inferno', number_of_features)
    for i in order:
        ax_timeseries.plot(
            time_window*1e3,
            data_ordered[i, :],
            label=f'{c["full_labels"][i]}' if c['full_labels'] is not None else f'Feature {i+1}',
            color=colors(i),
            alpha=1,
            zorder=100
        )
    mean_timeseries = data_ordered.mean(axis=0)
    std_timeseries = data_ordered.std(axis=0, ddof=1)/np.sqrt(number_of_features)
    _ = ax_timeseries.plot(
        time_window*1e3,
        mean_timeseries,
        color='green',
        label='Mean',
        zorder=130,
        linewidth=1.2
    )
    _ = ax_timeseries.fill_between(
        time_window*1e3,
        mean_timeseries - std_timeseries,
        mean_timeseries + std_timeseries,
        color='green',
        label='S.E.M',
        alpha=0.3,
        zorder=130,
        linewidth=1.2
    )
    # Legend into multiple columns if needed
    handles, labels = ax_timeseries.get_legend_handles_labels()
    ncol = int(np.ceil(len(labels) / 4))
    _ = ax_timeseries.legend(handles, labels, ncol=ncol, fontsize=7)
    ax_timeseries.grid(visible=True)

    # Avoid identical vmax or NaN
    vmax = np.abs(data_ordered).max()
    if not np.isfinite(vmax) or vmax == 0:
        raise ValueError("" \
        "Data contain non-finite values or identical min/max, cannot plot heatmap." \
        "trf_heatmap_plot aborted."
        )
    # Heatmap plot
    im = ax_heatmap.pcolormesh(
        time_window * 1e3, 
        np.arange(number_of_features), 
        data_ordered, 
        cmap='RdBu_r', 
        shading='auto',
        vmin=-vmax,
        vmax=vmax
    )
    # Axes labels and limits
    _ = ax_heatmap.set(
        xlabel='Time (ms)',
        ylabel=c['ylabel'] if c['ylabel'] is not None else 'Features',
        yticks=c['yticks'] if c['yticks'] is not None else np.arange(0, number_of_features, 1),
        yticklabels=c['yticklabels'] if c['yticklabels'] is not None else [str(i+1) for i in range(number_of_features)] 
    )
    # Create colorbar
    _ = fig.colorbar(
        im, 
        ax=ax_heatmap, 
        orientation='horizontal', 
        shrink=1, 
        label='Amplitude (a.u.)', 
        aspect=15
    )
    if show:
        plt.show(block=True)
    elif output_filepath is not None:
        _ = fig.savefig(
            output_filepath,
            dpi=dpi
        )
    plt.close(fig)    
    return True

@safe_plot
def topoplot(
    coefficient_values: np.ndarray,
    coefficient_name: str,
    info: mne.Info,
    figsize: tuple = (6, 5),
    figkwargs: dict = {},
    show: bool = False,
    colors:str = 'OrRd',
    output_filepath: Union[Path, str, None]=None,
    dpi: int = 300,
    logger: logging.Logger = None
)-> bool:
    """
    Plot topographic map of coefficient values.
    
    Parameters
    ----------
    coefficient_values : np.ndarray
        1D array of coefficient values for each channel.
    coefficient_name : str
        Name of the coefficient to display on the colorbar.
    info : mne.Info
        MNE Info object containing channel locations.
    figsize : tuple
        Size of the figure.
    figkwargs : dict
        Additional keyword arguments for the figure.
    show : bool
        Whether to display the figure. This overrides saving if True.
    colors : str
        Colormap to use for the topomap.
    output_filepath : Union[Path, str, None]
        Filepath to save the output figure. If None, the figure is not saved.
    dpi : int
        Dots per inch for the saved figure.
    logger : logging.Logger
        Logger for logging warnings or errors.
    
    Returns
    -------
    bool
        True if the figure was created successfully.
    """
    # Create figure and title
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        **figkwargs
    )
    
    # Make topomap
    im = mne.viz.plot_topomap(
        data=coefficient_values, 
        pos=info, 
        cmap=colors,
        vlim=(coefficient_values.min(), coefficient_values.max()),
        show=False, 
        sphere=0.07, 
        axes=ax,
        extrapolate='local',  # Esto limita el dibujo al área donde hay sensores
        border=0
    )
    
    # Avoid identical vmin/vmax or NaN
    vmin = coefficient_values.min()
    vmax = coefficient_values.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        if logger is not None:
            log_stage(
                f"Coefficient values contain non-finite values or identical min/max, topomap {coefficient_name} aborted.",
                level="WARNING",logger=logger
            )
        return False
    
    elif np.mean(np.abs(coefficient_values) < 0.001):
        if logger is not None:
            log_stage(
                f"Coefficient values are too close to zero (mean < 0.001), topomap {coefficient_name} aborted.",
                level="WARNING",logger=logger
            )

        return False
    cbar = plt.colorbar(
        im[0],
        ax=ax, 
        shrink=0.85,
        label=coefficient_name,
        orientation='horizontal',
        boundaries=np.linspace(vmin, vmax, 100) if vmin != vmax else None,
        ticks=np.linspace(vmin, vmax, 9) if vmin != vmax else [vmin]
    )

    # Format colorbar to 3 decimals, not scientific
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    cbar.update_ticks()
    
    if show:
        plt.show(block=True)
    elif output_filepath is not None:
        _ = fig.savefig(
            output_filepath,
            dpi=dpi
        )
    plt.close(fig)
    
    return True

# ==================
# Validation figures
@safe_plot
def hyperparameter_selection(
    alphas_grid: np.ndarray,
    correlations: np.ndarray,
    correlations_std: np.ndarray,
    correlations_train: np.ndarray,
    times: np.ndarray,
    trfs: np.ndarray,
    alpha_subject: float,
    correlation_limit_percentage: float,
    subject: int,
    attribute: str,
    band: str,
    save_path: str,
    overwrite: bool = True
):
    """
    Plot hyperparameter selection results including correlations, combined metrics, and TRFs.
    
    Parameters
    ----------
    alphas_grid : np.ndarray
        Array of alpha values that were swept.
    correlations : np.ndarray
        Array of correlation values on the test set.
    correlations_std : np.ndarray
        Array of standard deviations of the correlations.
    correlations_train : np.ndarray
        Array of correlation values on the training set.
    trfs : np.ndarray
        Array of Temporal Response Functions corresponding to each alpha.
    alpha_subject : float
        Selected alpha value for the subject.
    correlation_limit_percentage : float
        Percentage limit to highlight good correlation range.
    subject : int
        Subject number.
    attribute : str
        Stimulus type.
    band : str
        Frequency band.
    save_path : str
        Path to save the figure.
    overwrite : bool
        Whether to overwrite existing figures.
    
    Returns
    -------
    bool
        True if the figure was created successfully.
    """
    
    # Create figure with custom layout: 2x2 grid at top, 1x1 at bottom spanning both columns
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    # Top row: Correlation (left) and Correlation Ratio (right)
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_corr_ratio = fig.add_subplot(gs[0, 1])
    
    # Middle row: Combined Metric (left) and Combined Metric Normalized (right)
    ax_combined = fig.add_subplot(gs[1, 0])
    ax_combined_norm = fig.add_subplot(gs[1, 1])
    
    # Bottom row: TRFs spanning both columns
    ax_trfs = fig.add_subplot(gs[2, :])
    
    fig.suptitle(f'{band} - {attribute} - Subject {subject}', fontsize=16)
    
    # Find relevant range within correlation_limit_percentage
    relative_difference = abs((correlations.max() - correlations)/correlations.max())
    good_indexes_range = np.where(relative_difference < correlation_limit_percentage)[0]
    
    # Correlations plots
    ax_corr.plot(
        alphas_grid, correlations, 'o--', color='C0'
    )
    ax_corr.errorbar(
        alphas_grid, correlations, yerr=correlations_std, fmt='none', 
        ecolor='black', elinewidth=0.5, capsize=0.5
    )
    ax_corr.vlines(
        alphas_grid[correlations.argmax()], ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
        linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation'
    )
    ax_corr.vlines(
        alpha_subject, ax_corr.get_ylim()[0], ax_corr.get_ylim()[1], 
        linestyle='dashed', color='red', linewidth=1.5, label='Selected value'
    )
    if good_indexes_range.size > 1:
        ax_corr.axvspan(
            alphas_grid[good_indexes_range[0]], alphas_grid[good_indexes_range[-1]], 
            alpha=0.4, color='green', 
            label=f'{int(correlation_limit_percentage*100)}% of maximum correlation'
        )
    ax_corr.set(
        xlabel=r'Ridge parameter $\alpha$', ylabel='Mean correlation (Test)', 
        xscale='log', xlim=([alphas_grid[0], alphas_grid[-1]])
    )
    ax_corr.grid(visible=True)
    ax_corr.legend(fontsize=8)
    
    # Correlation Ratio plot
    ax_corr_ratio.plot(
        alphas_grid, 1e2*(correlations-correlations_train)/correlations_train, 'o--', color='C0'
    )
    ax_corr_ratio.vlines(
        alphas_grid[correlations.argmax()], ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
        linestyle='dashed', color='black', linewidth=1.5, label='Maximum correlation'
    )
    ax_corr_ratio.vlines(
        alpha_subject, ax_corr_ratio.get_ylim()[0], ax_corr_ratio.get_ylim()[1], 
        linestyle='dashed', color='red', linewidth=1.5, label='Selected value'
    )
    if good_indexes_range.size > 1:
        ax_corr_ratio.axvspan(
            alphas_grid[good_indexes_range[0]], alphas_grid[good_indexes_range[-1]], 
            alpha=0.4, color='green', 
            label=f'{int(correlation_limit_percentage*100)}% of maximum correlation'
        )
    ax_corr_ratio.set(
        xlabel=r'Ridge parameter $\alpha$', ylabel=r'Correlation (Test-Train)/Test[\%]', 
        xscale='log', xlim=([alphas_grid[0], alphas_grid[-1]])
    )
    ax_corr_ratio.grid(visible=True)
    ax_corr_ratio.legend(fontsize=8)
    
    # Combined Metric plot
    lambda2 = 1.5
    combined_metric = correlations - lambda2*np.maximum(0, correlations_train-correlations)
    ax_combined.plot(alphas_grid, combined_metric, 'o--', color='C2')
    combined_max_idx = combined_metric.argmax()
    ax_combined.vlines(
        alphas_grid[combined_max_idx], ax_combined.get_ylim()[0], ax_combined.get_ylim()[1], 
        linestyle='dashed', color='black', linewidth=1.5, label='Maximum combined metric'
    )
    ax_combined.vlines(
        alpha_subject, ax_combined.get_ylim()[0], ax_combined.get_ylim()[1], 
        linestyle='dashed', color='red', linewidth=1.5, label='Selected value'
    )
    if good_indexes_range.size > 1:
        ax_combined.axvspan(
            alphas_grid[good_indexes_range[0]], alphas_grid[good_indexes_range[-1]], 
            alpha=0.4, color='green', 
            label=f'{int(correlation_limit_percentage*100)}% of maximum correlation'
        )
    ax_combined.set(
        xlabel=r'Ridge parameter $\alpha$', 
        ylabel=rf'Combined Metric ($\lambda_2$={lambda2})', 
        xscale='log', xlim=([alphas_grid[0], alphas_grid[-1]])
    )
    ax_combined.grid(visible=True)
    ax_combined.legend(fontsize=8)
    
    # Combined Metric Normalized plot
    corr_norm = (correlations - correlations.min()) / (correlations.max() - correlations.min())
    corr_diff_norm = np.maximum(0, correlations_train-correlations)
    corr_diff_norm = corr_diff_norm / (corr_diff_norm.max() + 1e-10)
    combined_metric_norm = corr_norm - lambda2*corr_diff_norm
    ax_combined_norm.plot(alphas_grid, combined_metric_norm, 'o--', color='C3')
    combined_norm_max_idx = combined_metric_norm.argmax()
    ax_combined_norm.vlines(
        alphas_grid[combined_norm_max_idx], ax_combined_norm.get_ylim()[0], ax_combined_norm.get_ylim()[1], 
        linestyle='dashed', color='black', linewidth=1.5, label='Maximum normalized metric'
    )
    ax_combined_norm.vlines(
        alpha_subject, ax_combined_norm.get_ylim()[0], ax_combined_norm.get_ylim()[1], 
        linestyle='dashed', color='red', linewidth=1.5, label='Selected value'
    )
    if good_indexes_range.size > 1:
        ax_combined_norm.axvspan(
            alphas_grid[good_indexes_range[0]], alphas_grid[good_indexes_range[-1]], 
            alpha=0.4, color='green', 
            label=f'{int(correlation_limit_percentage*100)}% of maximum correlation'
        )
    ax_combined_norm.set(
        xlabel=r'Ridge parameter $\alpha$', 
        ylabel='Normalized Combined Metric', 
        xscale='log', xlim=([alphas_grid[0], alphas_grid[-1]])
    )
    ax_combined_norm.grid(visible=True)
    ax_combined_norm.legend(fontsize=8)
    
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(
        vmin=np.log10(alphas_grid.min()), 
        vmax=np.log10(alphas_grid.max())
    )
    for alpha, trf in zip(alphas_grid, trfs):
        color = cmap(norm(np.log10(alpha)))
        ax_trfs.plot(
            times*1e3,
            trf,
            color=color,
            linewidth=1.5,
            alpha=0.8
        )
    selected_idx = np.argmin(np.abs(alphas_grid - alpha_subject))
    ax_trfs.plot(
        times*1e3,
        trfs[selected_idx],
        color='black',
        linewidth=2.5,
        alpha=1.0,
        label=f'Selected α = {alpha_subject:.0f}' if alpha_subject >= 1 else f'Selected α = {alpha_subject:.3f}'
    )
    ax_trfs.set(
        xlabel='Time (ms)',
        ylabel='TRF Amplitude (a.u.)',
        ylim=(-np.abs(trfs[selected_idx]).max()*1.5, np.abs(trfs[selected_idx]).max()*1.5),
        title='Temporal Response Functions for Different Alpha Values'
    )
    ax_trfs.grid(True, alpha=0.3)
    ax_trfs.legend(loc='upper right', fontsize=10)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_trfs, orientation='vertical', pad=0.02, shrink=0.8, aspect=20)
    cbar.set_label(r'$\log_{10}(\alpha)$', fontsize=12)
    log_alphas = np.log10(alphas_grid)
    tick_positions = np.linspace(log_alphas.min(), log_alphas.max(), 5)
    cbar.set_ticks(tick_positions)
    alpha_values = [10**pos for pos in tick_positions]
    formatted_labels = []
    for alpha in alpha_values:
        if alpha < 0.01:
            formatted_labels.append(f'{alpha:.3f}')
        elif alpha < 1:
            formatted_labels.append(f'{alpha:.2f}')
        elif alpha < 1000:
            formatted_labels.append(f'{int(alpha)}')
        else:
            formatted_labels.append(f'{alpha:.1e}')
    cbar.set_ticklabels(formatted_labels)
    
    # Save figure
    if overwrite or (not Path(save_path).exists()):
        fig.savefig(
            save_path,
            dpi=300
        )
    return True