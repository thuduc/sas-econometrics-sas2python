"""Visualization utilities for Hidden Markov Models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple, Any, Dict
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
import warnings


def plot_states_timeline(
    states: np.ndarray,
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    state_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "Hidden States Timeline",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the sequence of hidden states over time.
    
    Parameters
    ----------
    states : array-like, shape (n_samples,)
        Sequence of states
    time_index : array-like, optional
        Time index for x-axis
    state_names : list of str, optional
        Names for each state
    figsize : tuple, default=(12, 4)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    states = np.asarray(states)
    n_states = len(np.unique(states))
    
    if time_index is None:
        time_index = np.arange(len(states))
        
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
        
    # Create color map
    colors = plt.cm.Set1(np.linspace(0, 1, n_states))
    
    # Plot states as colored regions
    for i in range(n_states):
        mask = states == i
        ax.fill_between(time_index, 0, 1, where=mask, 
                       alpha=0.7, color=colors[i], label=state_names[i])
        
    ax.set_ylim(0, 1)
    ax.set_xlim(time_index[0], time_index[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yticks([])
    
    return ax


def plot_state_probabilities(
    probabilities: np.ndarray,
    time_index: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    state_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "State Probabilities Over Time",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot posterior probabilities of states over time.
    
    Parameters
    ----------
    probabilities : array-like, shape (n_samples, n_states)
        Posterior probabilities
    time_index : array-like, optional
        Time index for x-axis
    state_names : list of str, optional
        Names for each state
    figsize : tuple, default=(12, 6)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    probabilities = np.asarray(probabilities)
    n_samples, n_states = probabilities.shape
    
    if time_index is None:
        time_index = np.arange(n_samples)
        
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
        
    # Plot stacked area chart
    ax.stackplot(time_index, probabilities.T, labels=state_names, alpha=0.7)
    
    ax.set_xlim(time_index[0], time_index[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return ax


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    state_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Transition Matrix",
    ax: Optional[plt.Axes] = None,
    annot: bool = True,
    fmt: str = ".3f",
    cmap: str = "Blues"
) -> plt.Axes:
    """
    Plot the transition matrix as a heatmap.
    
    Parameters
    ----------
    transition_matrix : array-like, shape (n_states, n_states)
        Transition probability matrix
    state_names : list of str, optional
        Names for each state
    figsize : tuple, default=(8, 6)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
    annot : bool, default=True
        Whether to annotate cells with values
    fmt : str, default=".3f"
        Format for annotations
    cmap : str, default="Blues"
        Colormap name
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    transition_matrix = np.asarray(transition_matrix)
    n_states = transition_matrix.shape[0]
    
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
        
    # Create heatmap
    sns.heatmap(transition_matrix, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': 'Probability'}, ax=ax,
                vmin=0, vmax=1, square=True)
    
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title)
    
    return ax


def plot_gaussian_emissions(
    means: np.ndarray,
    covariances: np.ndarray,
    data: Optional[np.ndarray] = None,
    states: Optional[np.ndarray] = None,
    n_std: float = 2,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Gaussian Emission Distributions",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot Gaussian emission distributions for 2D data.
    
    Parameters
    ----------
    means : array-like, shape (n_states, 2)
        Mean vectors for each state
    covariances : array-like, shape (n_states, 2, 2)
        Covariance matrices for each state
    data : array-like, shape (n_samples, 2), optional
        Data points to overlay
    states : array-like, shape (n_samples,), optional
        State assignments for data points
    n_std : float, default=2
        Number of standard deviations for ellipse
    figsize : tuple, default=(10, 8)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    means = np.asarray(means)
    n_states, n_features = means.shape
    
    if n_features != 2:
        raise ValueError("This function only supports 2D data")
        
    colors = plt.cm.Set1(np.linspace(0, 1, n_states))
    
    # Plot data points if provided
    if data is not None:
        data = np.asarray(data)
        if states is not None:
            states = np.asarray(states)
            for i in range(n_states):
                mask = states == i
                if np.any(mask):
                    ax.scatter(data[mask, 0], data[mask, 1], 
                             c=[colors[i]], alpha=0.5, s=30,
                             label=f"State {i} data")
        else:
            ax.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.3, s=30)
            
    # Plot Gaussian ellipses
    for i in range(n_states):
        mean = means[i]
        cov = covariances[i]
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Width and height are 2 * sqrt(eigenvalue) * n_std
        width, height = 2 * np.sqrt(eigenvalues) * n_std
        
        # Create ellipse
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=colors[i], alpha=0.3,
                         edgecolor=colors[i], linewidth=2,
                         label=f"State {i}")
        ax.add_patch(ellipse)
        
        # Plot mean
        ax.scatter(mean[0], mean[1], c=[colors[i]], s=200, 
                  marker='x', linewidths=3)
        
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_discrete_emissions(
    emission_probs: np.ndarray,
    symbol_names: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Discrete Emission Probabilities",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot discrete emission probabilities as grouped bar chart.
    
    Parameters
    ----------
    emission_probs : array-like, shape (n_states, n_symbols)
        Emission probability matrix
    symbol_names : list of str, optional
        Names for each symbol
    state_names : list of str, optional
        Names for each state
    figsize : tuple, default=(10, 6)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    emission_probs = np.asarray(emission_probs)
    n_states, n_symbols = emission_probs.shape
    
    if symbol_names is None:
        symbol_names = [f"Symbol {i}" for i in range(n_symbols)]
        
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
        
    # Create grouped bar chart
    x = np.arange(n_symbols)
    width = 0.8 / n_states
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_states))
    
    for i in range(n_states):
        offset = (i - n_states/2 + 0.5) * width
        ax.bar(x + offset, emission_probs[i], width, 
               label=state_names[i], color=colors[i])
        
    ax.set_xlabel("Symbols")
    ax.set_ylabel("Emission Probability")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(symbol_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    return ax


def plot_model_selection(
    n_states_range: List[int],
    scores: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Selection Criteria",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot model selection criteria (AIC, BIC, etc.) vs number of states.
    
    Parameters
    ----------
    n_states_range : list of int
        Range of number of states tested
    scores : dict
        Dictionary mapping criterion names to lists of scores
    figsize : tuple, default=(10, 6)
        Figure size
    title : str
        Plot title
    ax : matplotlib Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    for criterion, values in scores.items():
        ax.plot(n_states_range, values, marker='o', label=criterion)
        
        # Mark minimum
        min_idx = np.argmin(values)
        ax.scatter(n_states_range[min_idx], values[min_idx], 
                  s=200, marker='*', c='red', zorder=5)
        ax.annotate(f'Min {criterion}\nn_states={n_states_range[min_idx]}',
                   xy=(n_states_range[min_idx], values[min_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
    ax.set_xlabel("Number of States")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_hmm_summary(
    model: Any,
    data: np.ndarray,
    states: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Create a comprehensive summary plot for an HMM model.
    
    Parameters
    ----------
    model : HMM model
        Fitted HMM model with results
    data : array-like
        Original data
    states : array-like, optional
        True states if available
    figsize : tuple, default=(15, 12)
        Figure size
        
    Returns
    -------
    fig : matplotlib Figure
        The figure containing all subplots
    """
    fig = plt.figure(figsize=figsize)
    
    # Get model results
    results = model.results
    predicted_states = results.states
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Plot 1: States timeline
    ax1 = fig.add_subplot(gs[0, :])
    plot_states_timeline(predicted_states, ax=ax1, title="Predicted States Timeline")
    
    # Plot 2: State probabilities
    if results.posterior_probabilities is not None:
        ax2 = fig.add_subplot(gs[1, :])
        plot_state_probabilities(results.posterior_probabilities, ax=ax2)
        
    # Plot 3: Transition matrix
    ax3 = fig.add_subplot(gs[2, 0])
    plot_transition_matrix(results.transition_matrix, ax=ax3)
    
    # Plot 4: Model-specific visualization
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Add model information
    info_text = f"Model: {model.__class__.__name__}\n"
    info_text += f"Number of states: {model.n_states}\n"
    info_text += f"Log-likelihood: {results.log_likelihood:.2f}\n"
    info_text += f"AIC: {results.aic:.2f}\n" if results.aic else ""
    info_text += f"BIC: {results.bic:.2f}\n" if results.bic else ""
    info_text += f"Converged: {results.converged}\n"
    info_text += f"Iterations: {results.n_iterations}"
    
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title("Model Summary")
    
    fig.suptitle(f"{model.__class__.__name__} Analysis", fontsize=16)
    
    return fig


def animate_hmm_sequence(
    data: np.ndarray,
    states: np.ndarray,
    interval: int = 100,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Any:
    """
    Create an animation of HMM sequence evolution.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Data sequence
    states : array-like, shape (n_samples,)
        State sequence
    interval : int, default=100
        Milliseconds between frames
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save animation
        
    Returns
    -------
    anim : matplotlib animation
        The animation object
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        warnings.warn("Animation requires matplotlib.animation")
        return None
        
    data = np.asarray(data)
    states = np.asarray(states)
    n_samples = len(data)
    n_states = len(np.unique(states))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Initialize plots
    colors = plt.cm.Set1(np.linspace(0, 1, n_states))
    
    # Data plot
    if data.ndim == 1:
        line, = ax1.plot([], [], 'b-', alpha=0.5)
        scatter = ax1.scatter([], [], c=[], s=50)
        ax1.set_xlim(0, n_samples)
        ax1.set_ylim(data.min() - 0.1, data.max() + 0.1)
    else:
        scatter = ax1.scatter([], [], c=[], s=50)
        ax1.set_xlim(data[:, 0].min() - 0.1, data[:, 0].max() + 0.1)
        ax1.set_ylim(data[:, 1].min() - 0.1, data[:, 1].max() + 0.1)
        
    ax1.set_title("Data Sequence")
    
    # State plot
    state_bars = ax2.bar(range(n_states), [0] * n_states, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("State")
    ax2.set_ylabel("Current State")
    ax2.set_title("Current State")
    
    def init():
        if data.ndim == 1:
            line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        for bar in state_bars:
            bar.set_height(0)
        return [scatter] + list(state_bars)
        
    def animate(frame):
        # Update data plot
        if data.ndim == 1:
            line.set_data(range(frame + 1), data[:frame + 1])
            scatter.set_offsets([[frame, data[frame]]])
        else:
            scatter.set_offsets(data[:frame + 1])
            
        scatter.set_color([colors[states[frame]]])
        
        # Update state plot
        for i, bar in enumerate(state_bars):
            bar.set_height(1 if states[frame] == i else 0)
            
        return [scatter] + list(state_bars)
        
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_samples,
                        interval=interval, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
        
    plt.close()
    
    return anim