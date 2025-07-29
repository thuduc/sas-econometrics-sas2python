"""
Time Series Clustering with HMM - Python implementation of hmmex02.sas.

This example demonstrates using HMM for clustering time series data based on
their dynamic patterns. It shows how HMM can identify different behavioral
patterns in multiple time series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import warnings

from src.models import GaussianHMM, GaussianMixtureHMM
from src.utils.visualization import (
    plot_states_timeline,
    plot_transition_matrix,
    plot_model_selection
)


def generate_synthetic_time_series(
    n_series: int = 100,
    n_timepoints: int = 100,
    n_patterns: int = 3,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic time series with different patterns.
    
    Parameters
    ----------
    n_series : int
        Number of time series to generate
    n_timepoints : int
        Length of each time series
    n_patterns : int
        Number of different patterns
    noise_level : float
        Standard deviation of noise to add
    seed : int
        Random seed
        
    Returns
    -------
    data : array, shape (n_series * n_timepoints, n_features)
        Flattened time series data
    true_labels : array, shape (n_series,)
        True pattern labels for each series
    pattern_names : list
        Names of the patterns
    """
    np.random.seed(seed)
    
    # Define patterns
    t = np.linspace(0, 4*np.pi, n_timepoints)
    patterns = []
    pattern_names = []
    
    if n_patterns >= 1:
        # Pattern 1: Trending upward with small oscillations
        pattern1 = 0.5 * t / (4*np.pi) + 0.1 * np.sin(4*t)
        patterns.append(pattern1)
        pattern_names.append("Upward Trend")
        
    if n_patterns >= 2:
        # Pattern 2: Oscillating with increasing amplitude
        pattern2 = 0.3 * np.sin(2*t) * (1 + 0.1 * t)
        patterns.append(pattern2)
        pattern_names.append("Growing Oscillation")
        
    if n_patterns >= 3:
        # Pattern 3: Step changes
        pattern3 = np.zeros_like(t)
        pattern3[t < np.pi] = 0.2
        pattern3[(t >= np.pi) & (t < 2*np.pi)] = -0.3
        pattern3[(t >= 2*np.pi) & (t < 3*np.pi)] = 0.4
        pattern3[t >= 3*np.pi] = -0.1
        patterns.append(pattern3)
        pattern_names.append("Step Changes")
    
    # Generate time series
    data = []
    true_labels = []
    series_per_pattern = n_series // n_patterns
    
    for pattern_idx, pattern in enumerate(patterns):
        for _ in range(series_per_pattern):
            # Add random scaling and shift
            scale = np.random.uniform(0.8, 1.2)
            shift = np.random.uniform(-0.2, 0.2)
            
            # Generate series
            series = scale * pattern + shift + noise_level * np.random.randn(n_timepoints)
            
            # Add some random walk component
            random_walk = np.cumsum(0.01 * np.random.randn(n_timepoints))
            series += random_walk
            
            data.extend(series)
            true_labels.append(pattern_idx)
    
    # Handle remaining series
    remaining = n_series - len(true_labels)
    for _ in range(remaining):
        pattern_idx = np.random.randint(n_patterns)
        pattern = patterns[pattern_idx]
        scale = np.random.uniform(0.8, 1.2)
        shift = np.random.uniform(-0.2, 0.2)
        series = scale * pattern + shift + noise_level * np.random.randn(n_timepoints)
        series += np.cumsum(0.01 * np.random.randn(n_timepoints))
        data.extend(series)
        true_labels.append(pattern_idx)
    
    data = np.array(data).reshape(-1, 1)
    true_labels = np.array(true_labels)
    
    return data, true_labels, pattern_names


def create_feature_matrix(
    data: np.ndarray,
    n_series: int,
    n_timepoints: int,
    window_size: int = 10
) -> np.ndarray:
    """
    Create feature matrix from time series using sliding windows.
    
    Parameters
    ----------
    data : array, shape (n_series * n_timepoints, 1)
        Flattened time series data
    n_series : int
        Number of time series
    n_timepoints : int
        Length of each series
    window_size : int
        Size of sliding window for features
        
    Returns
    -------
    features : array, shape (n_samples, n_features)
        Feature matrix
    """
    features = []
    
    for i in range(n_series):
        series = data[i*n_timepoints:(i+1)*n_timepoints, 0]
        
        # Extract features using sliding windows
        for j in range(0, n_timepoints - window_size + 1, window_size // 2):
            window = series[j:j+window_size]
            
            # Calculate features
            feat = [
                np.mean(window),                    # Mean
                np.std(window),                      # Standard deviation
                np.max(window) - np.min(window),     # Range
                window[-1] - window[0],              # Trend
                np.mean(np.abs(np.diff(window))),   # Mean absolute difference
            ]
            
            features.append(feat)
    
    return np.array(features)


def cluster_time_series_hmm(
    data: np.ndarray,
    n_series: int,
    n_timepoints: int,
    n_states: int = 3,
    window_size: int = 10
) -> Tuple[GaussianHMM, np.ndarray, np.ndarray]:
    """
    Cluster time series using HMM.
    
    Parameters
    ----------
    data : array
        Time series data
    n_series : int
        Number of series
    n_timepoints : int
        Length of each series
    n_states : int
        Number of HMM states
    window_size : int
        Window size for feature extraction
        
    Returns
    -------
    model : GaussianHMM
        Fitted HMM model
    features : array
        Feature matrix
    cluster_labels : array
        Cluster assignment for each series
    """
    # Create feature matrix
    features = create_feature_matrix(data, n_series, n_timepoints, window_size)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Calculate lengths for each series
    n_windows_per_series = (n_timepoints - window_size) // (window_size // 2) + 1
    lengths = [n_windows_per_series] * n_series
    
    # Fit HMM
    model = GaussianHMM(n_states=n_states, covariance_type='full', random_state=42)
    model.fit(features_scaled, lengths=lengths)
    
    # Get state sequences
    states = model.predict(features_scaled, lengths=lengths)
    
    # Assign each series to a cluster based on most common state
    cluster_labels = []
    start_idx = 0
    
    for length in lengths:
        series_states = states[start_idx:start_idx+length]
        # Most common state
        cluster = np.argmax(np.bincount(series_states))
        cluster_labels.append(cluster)
        start_idx += length
    
    cluster_labels = np.array(cluster_labels)
    
    return model, features_scaled, cluster_labels


def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pattern_names: List[str]
) -> Dict[str, float]:
    """
    Evaluate clustering performance.
    
    Parameters
    ----------
    true_labels : array
        True cluster labels
    pred_labels : array
        Predicted cluster labels
    pattern_names : list
        Names of patterns
        
    Returns
    -------
    metrics : dict
        Evaluation metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.stats import mode
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # Calculate purity
    n_clusters = len(np.unique(pred_labels))
    n_true = len(np.unique(true_labels))
    
    # Map predicted clusters to true clusters
    mapping = {}
    purity = 0
    
    for pred_cluster in range(n_clusters):
        mask = pred_labels == pred_cluster
        if np.any(mask):
            true_in_cluster = true_labels[mask]
            most_common = mode(true_in_cluster).mode[0]
            mapping[pred_cluster] = most_common
            purity += np.sum(true_in_cluster == most_common)
    
    purity /= len(true_labels)
    
    print("\nClustering Evaluation:")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    print(f"Purity: {purity:.3f}")
    
    print("\nCluster Mapping:")
    for pred, true in mapping.items():
        print(f"Cluster {pred} -> {pattern_names[true]}")
    
    return {
        'ari': ari,
        'nmi': nmi,
        'purity': purity,
        'mapping': mapping
    }


def visualize_clustering_results(
    data: np.ndarray,
    n_series: int,
    n_timepoints: int,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pattern_names: List[str],
    model: GaussianHMM,
    features: np.ndarray
) -> plt.Figure:
    """Create comprehensive visualization of clustering results."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Sample time series by true pattern
    ax1 = fig.add_subplot(gs[0, :2])
    colors = plt.cm.Set1(np.linspace(0, 1, len(pattern_names)))
    
    for pattern_idx, pattern_name in enumerate(pattern_names):
        # Find series with this pattern
        series_indices = np.where(true_labels == pattern_idx)[0]
        
        # Plot up to 5 examples
        for i, idx in enumerate(series_indices[:5]):
            series = data[idx*n_timepoints:(idx+1)*n_timepoints, 0]
            time = np.arange(len(series))
            
            if i == 0:
                ax1.plot(time, series, color=colors[pattern_idx], 
                        alpha=0.7, label=pattern_name)
            else:
                ax1.plot(time, series, color=colors[pattern_idx], alpha=0.5)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Sample Time Series by True Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transition matrix
    ax2 = fig.add_subplot(gs[0, 2])
    plot_transition_matrix(model.model.transmat_, ax=ax2)
    
    # Plot 3: Confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create confusion matrix
    n_true = len(pattern_names)
    n_pred = len(np.unique(pred_labels))
    conf_matrix = np.zeros((n_true, n_pred))
    
    for i in range(len(true_labels)):
        conf_matrix[true_labels[i], pred_labels[i]] += 1
    
    # Normalize by row
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    
    im = ax3.imshow(conf_matrix, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(n_pred))
    ax3.set_yticks(range(n_true))
    ax3.set_xticklabels([f'C{i}' for i in range(n_pred)])
    ax3.set_yticklabels(pattern_names)
    ax3.set_xlabel('Predicted Cluster')
    ax3.set_ylabel('True Pattern')
    ax3.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(n_true):
        for j in range(n_pred):
            text = ax3.text(j, i, f'{conf_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if conf_matrix[i, j] < 0.5 else "white")
    
    # Plot 4: Feature space (first 2 PCA components)
    ax4 = fig.add_subplot(gs[1, 1:])
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Calculate series centers in feature space
    n_windows_per_series = len(features) // n_series
    series_centers = []
    
    for i in range(n_series):
        start = i * n_windows_per_series
        end = (i + 1) * n_windows_per_series
        center = features_2d[start:end].mean(axis=0)
        series_centers.append(center)
    
    series_centers = np.array(series_centers)
    
    # Plot by predicted clusters
    for cluster in range(n_pred):
        mask = pred_labels == cluster
        ax4.scatter(series_centers[mask, 0], series_centers[mask, 1],
                   c=[colors[cluster]], label=f'Cluster {cluster}',
                   s=100, alpha=0.7)
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    ax4.set_title('Time Series in Feature Space (PCA)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Examples from each cluster
    for cluster_idx in range(min(3, n_pred)):
        ax = fig.add_subplot(gs[2, cluster_idx])
        
        # Find series in this cluster
        cluster_series = np.where(pred_labels == cluster_idx)[0]
        
        # Plot up to 10 examples
        for i, idx in enumerate(cluster_series[:10]):
            series = data[idx*n_timepoints:(idx+1)*n_timepoints, 0]
            ax.plot(series, alpha=0.5, color=colors[cluster_idx])
        
        ax.set_title(f'Cluster {cluster_idx} Examples')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Clustering Results', fontsize=16)
    
    return fig


def model_selection_experiment(
    data: np.ndarray,
    n_series: int,
    n_timepoints: int,
    max_states: int = 10
) -> Dict[str, List[float]]:
    """
    Perform model selection to find optimal number of states.
    
    Parameters
    ----------
    data : array
        Time series data
    n_series : int
        Number of series
    n_timepoints : int
        Length of each series
    max_states : int
        Maximum number of states to try
        
    Returns
    -------
    scores : dict
        AIC and BIC scores for each model
    """
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    
    # Create features
    features = create_feature_matrix(data, n_series, n_timepoints)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Calculate lengths
    n_windows_per_series = len(features) // n_series
    lengths = [n_windows_per_series] * n_series
    
    # Try different numbers of states
    n_states_range = range(2, max_states + 1)
    aic_scores = []
    bic_scores = []
    
    for n_states in n_states_range:
        print(f"\nFitting model with {n_states} states...")
        
        try:
            model = GaussianHMM(n_states=n_states, covariance_type='diag', 
                              random_state=42, n_iter=100)
            model.fit(features_scaled, lengths=lengths)
            
            # Get scores
            results = model.results
            aic_scores.append(results.aic)
            bic_scores.append(results.bic)
            
            print(f"  AIC: {results.aic:.2f}")
            print(f"  BIC: {results.bic:.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            aic_scores.append(np.inf)
            bic_scores.append(np.inf)
    
    # Find optimal number of states
    best_aic_idx = np.argmin(aic_scores)
    best_bic_idx = np.argmin(bic_scores)
    
    print(f"\nOptimal number of states:")
    print(f"  By AIC: {n_states_range[best_aic_idx]}")
    print(f"  By BIC: {n_states_range[best_bic_idx]}")
    
    return {
        'AIC': aic_scores,
        'BIC': bic_scores
    }


def main():
    """Run the time series clustering example."""
    # Generate synthetic data
    n_series = 150
    n_timepoints = 100
    n_patterns = 3
    
    print("Generating synthetic time series data...")
    data, true_labels, pattern_names = generate_synthetic_time_series(
        n_series=n_series,
        n_timepoints=n_timepoints,
        n_patterns=n_patterns,
        noise_level=0.15
    )
    
    print(f"Generated {n_series} time series with {n_patterns} patterns")
    print(f"Patterns: {', '.join(pattern_names)}")
    
    # Model selection
    scores = model_selection_experiment(data, n_series, n_timepoints, max_states=8)
    
    # Plot model selection results
    fig1, ax = plt.subplots(figsize=(10, 6))
    plot_model_selection(list(range(2, len(scores['AIC']) + 2)), scores, ax=ax)
    plt.show()
    
    # Cluster with optimal number of states
    optimal_states = list(range(2, len(scores['BIC']) + 2))[np.argmin(scores['BIC'])]
    print(f"\nClustering with {optimal_states} states...")
    
    model, features, pred_labels = cluster_time_series_hmm(
        data, n_series, n_timepoints, n_states=optimal_states
    )
    
    # Evaluate clustering
    metrics = evaluate_clustering(true_labels, pred_labels, pattern_names)
    
    # Visualize results
    fig2 = visualize_clustering_results(
        data, n_series, n_timepoints,
        true_labels, pred_labels, pattern_names,
        model, features
    )
    plt.show()
    
    # Advanced: Try Gaussian Mixture HMM
    print("\n" + "=" * 60)
    print("GAUSSIAN MIXTURE HMM")
    print("=" * 60)
    
    # Create more complex features
    features_complex = []
    for i in range(n_series):
        series = data[i*n_timepoints:(i+1)*n_timepoints, 0]
        
        # Use entire series as one long feature vector (downsampled)
        downsampled = series[::5]  # Take every 5th point
        features_complex.append(downsampled)
    
    features_complex = np.array(features_complex)
    
    # Fit Gaussian Mixture HMM
    gmm_model = GaussianMixtureHMM(
        n_states=2, 
        n_components=2,
        covariance_type='diag',
        random_state=42
    )
    
    # Each series is one sequence
    gmm_model.fit(features_complex)
    
    # Get cluster assignments
    gmm_states = gmm_model.predict(features_complex)
    
    print("\nGaussian Mixture HMM Results:")
    print(f"State distribution: {np.bincount(gmm_states)}")
    
    # Simple accuracy (most common state per true pattern)
    gmm_accuracy = 0
    for pattern in range(n_patterns):
        mask = true_labels == pattern
        if np.any(mask):
            pattern_states = gmm_states[mask]
            most_common = np.argmax(np.bincount(pattern_states))
            gmm_accuracy += np.sum(pattern_states == most_common)
    
    gmm_accuracy /= len(true_labels)
    print(f"Simple accuracy: {gmm_accuracy:.3f}")
    
    print("\n" + "=" * 60)
    print("Time Series Clustering Analysis Complete!")
    print("=" * 60)
    
    return {
        'model': model,
        'gmm_model': gmm_model,
        'data': data,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'metrics': metrics
    }


if __name__ == "__main__":
    results = main()