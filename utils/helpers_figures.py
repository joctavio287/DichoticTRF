"""
Helpers for plotting figures
"""
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import cm
from pathlib import Path
from typing import Union
import numpy as np
import mne

import config

if config.USE_SCIENCE_PLOTS:
    import scienceplots
    plt.style.use("science")


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
    None
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
        fig.savefig(
            output_filepath, dpi=dpi
        )
    plt.close(fig)

    if verbose:
        print('\n\t Figure saved to: ', output_filepath)

def evoked_potential_plot(
    evoked: mne.Evoked,
    output_filepath: Union[Path, str, None]=None,
    time_window: Union[tuple, list, None]=None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    show: bool = False,
    figkwargs: dict = {},
    verbose: bool = True
)-> None:
    """
    Plot evoked potential with mean ERP.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data to plot.
    output_filepath : Union[Path, str, None]
        Filepath to save the output figure. If None, the figure is not saved.
    time_window : Union[tuple, list, None]
        Time window (start, end) in seconds for x-axis ticks. If None, uses full range of evoked.times
    figsize : tuple
        Size of the figure.
    dpi : int
        Dots per inch for the saved figure.
    show : bool
        Whether to display the figure. This overrides saving if True.
    figkwargs : dict
        Additional keyword arguments for the figure.
    """
    
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        **figkwargs
    )
    
    evoked_plot = evoked.plot(
        scalings={'eeg':1},
        zorder='std',
        time_unit='ms',
        show=False,
        spatial_colors=True,
        axes=ax,
        gfp=True
    )
    mean_plot = ax.plot(
        evoked.times*1e3, #ms
        evoked._data.mean(axis=0),
        color='black',
        label='Mean ERP',
        zorder=130,
        linewidth=1.2
    )
    # evoked_plot = evoked.plot_joint( # same plot but with topomap at specific times
    #     times='peaks',
    #     show=True,
    #     title='Evoked Potential - Bips',
    #     ts_args=dict(time_unit='ms')
    # )

    # Eliminar la etiqueta "Nave"
    for txt in fig.findobj(mtext.Text):
        if "ave" in txt.get_text():
                txt.remove()
    
    maximum = np.max(np.abs(evoked._data)) 
    _ = ax.set_ylim(-maximum, maximum)
    _ = ax.set_xlabel('Time (ms)')

    time_window = time_window if time_window is not None else (evoked.times[0], evoked.times[-1])
    _ = ax.set_xticks(
        np.arange(
            time_window[0] * 1e3,
            time_window[1] * 1e3 + 1,
            100
        )
    )
    _ = ax.legend(loc='upper right', fontsize=8)
    _ = ax.grid(True)

    if show:
        plt.show(block=True)
    elif output_filepath is not None:
        fig.savefig(
            output_filepath,
            dpi=dpi
        )
    plt.close(fig)
    
    if verbose:
        print('\n\t Figure saved to: ', output_filepath)

# ==================
# Validation figures
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
    None
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