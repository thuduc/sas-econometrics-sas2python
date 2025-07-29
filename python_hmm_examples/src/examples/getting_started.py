"""
Getting Started Example for HMM - Python implementation of hmmgs.sas.

This example demonstrates:
1. Basic Gaussian HMM with 2D observations
2. Time series clustering with multiple volatility states  
3. Gaussian Mixture HMM with multiple components per state
4. Regime-switching regression and AR models
5. Discrete HMM for finite state observations
6. Poisson HMM for count data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

from src.models import (
    GaussianHMM, 
    GaussianMixtureHMM, 
    DiscreteHMM,
    PoissonHMM,
    RegimeSwitchingAR,
    RegimeSwitchingRegression
)
from src.utils.visualization import (
    plot_states_timeline,
    plot_gaussian_emissions,
    plot_transition_matrix,
    plot_hmm_summary
)


def generate_gaussian_hmm_data(
    n_samples: int = 10000,
    seed: int = 1234
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic data for Gaussian HMM example."""
    np.random.seed(seed)
    
    # Parameters from SAS example
    pi1 = 0.5  # Initial probability for state 1
    a11 = 0.001  # Transition prob from state 1 to 1
    a22 = 0.001  # Transition prob from state 2 to 2
    
    # State 1: mean=[1,1], cov=I
    mu1 = np.array([1.0, 1.0])
    sigma1 = np.eye(2)
    
    # State 2: mean=[-1,-1], cov=I  
    mu2 = np.array([-1.0, -1.0])
    sigma2 = np.eye(2)
    
    # Generate states
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0 if np.random.rand() < pi1 else 1
    
    for t in range(1, n_samples):
        if states[t-1] == 0:
            states[t] = 0 if np.random.rand() < a11 else 1
        else:
            states[t] = 1 if np.random.rand() < a22 else 0
            
    # Generate observations
    X = np.zeros((n_samples, 2))
    for t in range(n_samples):
        if states[t] == 0:
            X[t] = np.random.multivariate_normal(mu1, sigma1)
        else:
            X[t] = np.random.multivariate_normal(mu2, sigma2)
            
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['t'] = range(n_samples)
    
    return df, states


def example_1_basic_gaussian_hmm():
    """Example 1: Basic Gaussian HMM."""
    print("=" * 60)
    print("Example 1: Basic Gaussian HMM")
    print("=" * 60)
    
    # Generate data
    df, true_states = generate_gaussian_hmm_data()
    X = df[['x', 'y']].values
    
    # Fit HMM
    model = GaussianHMM(n_states=2, covariance_type='full', random_state=42)
    model.fit(X)
    
    # Apply label switching to match true states
    model.label_switch(sort_by='means', ascending=False)
    
    # Get results
    results = model.results
    predicted_states = results.states
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == true_states)
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        
    print(f"\nModel Summary:")
    print(f"Log-likelihood: {results.log_likelihood:.2f}")
    print(f"AIC: {results.aic:.2f}")
    print(f"BIC: {results.bic:.2f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    print(f"\nTransition Matrix:")
    print(results.transition_matrix)
    
    print(f"\nState Means:")
    print(model.means_)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time series
    axes[0, 0].plot(df['t'][:1000], df['x'][:1000], alpha=0.7)
    axes[0, 0].set_title('X Time Series (first 1000 points)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('X')
    
    axes[0, 1].plot(df['t'][:1000], df['y'][:1000], alpha=0.7, color='orange')
    axes[0, 1].set_title('Y Time Series (first 1000 points)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Y')
    
    # Plot 2: Scatter plot with states
    plot_gaussian_emissions(
        model.means_, 
        model.covars_,
        data=X[:2000],
        states=predicted_states[:2000],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Gaussian Emissions (first 2000 points)')
    
    # Plot 3: Transition matrix
    plot_transition_matrix(results.transition_matrix, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return model, df


def example_2_time_series_clustering():
    """Example 2: Time series clustering with different volatility states."""
    print("\n" + "=" * 60)
    print("Example 2: Time Series Clustering")
    print("=" * 60)
    
    # Parameters
    n_sections = 100
    T = 100
    n_states = 6
    seed = 1234
    
    # Generate data with different volatility states
    np.random.seed(seed)
    
    # Means and variances for each state
    means = [0, 0, 0, 0, 0, 0]
    sigmas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    self_trans_prob = 0.95
    
    data = []
    true_states = []
    
    for sec in range(n_sections):
        # Random initial state
        state = np.random.randint(0, n_states)
        
        for t in range(T):
            # Generate observation
            y = means[state] + np.sqrt(sigmas[state]) * np.random.randn()
            data.append({'sec': sec, 't': t, 'y': y})
            true_states.append(state)
            
            # State transition
            if np.random.rand() > self_trans_prob:
                # Change state
                state = (state + np.random.randint(1, n_states)) % n_states
                
    df = pd.DataFrame(data)
    
    # Fit HMM
    model = GaussianHMM(n_states=6, covariance_type='spherical', random_state=42)
    
    # Create lengths array for multiple sequences
    lengths = [T] * n_sections
    model.fit(df[['y']].values, lengths=lengths)
    
    # Sort states by variance
    model.label_switch(sort_by='variance', ascending=True)
    
    # Get results
    results = model.results
    
    print(f"\nModel Summary:")
    print(f"Log-likelihood: {results.log_likelihood:.2f}")
    print(f"Converged: {results.converged}")
    
    print(f"\nEstimated Means:")
    print(model.means_.flatten())
    
    print(f"\nEstimated Standard Deviations:")
    print(np.sqrt(model.covars_.flatten()))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sample time series from different sections
    for i in range(3):
        section_data = df[df['sec'] == i]
        axes[0, 0].plot(section_data['t'], section_data['y'], 
                       alpha=0.7, label=f'Section {i}')
    axes[0, 0].set_title('Sample Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    
    # Plot 2: Histogram of all data
    axes[0, 1].hist(df['y'], bins=50, alpha=0.7, density=True)
    axes[0, 1].set_title('Distribution of Observations')
    axes[0, 1].set_xlabel('Y')
    axes[0, 1].set_ylabel('Density')
    
    # Plot 3: Estimated state means and stds
    states_idx = np.arange(n_states)
    axes[1, 0].errorbar(states_idx, model.means_.flatten(),
                       yerr=np.sqrt(model.covars_.flatten()),
                       fmt='o', capsize=5)
    axes[1, 0].set_title('Estimated State Parameters')
    axes[1, 0].set_xlabel('State')
    axes[1, 0].set_ylabel('Mean ± Std')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: State timeline for first section
    first_section_states = results.states[:T]
    plot_states_timeline(first_section_states, ax=axes[1, 1])
    axes[1, 1].set_title('States for First Section')
    
    plt.tight_layout()
    plt.show()
    
    return model, df


def example_3_gaussian_mixture_hmm():
    """Example 3: Gaussian Mixture HMM."""
    print("\n" + "=" * 60)
    print("Example 3: Gaussian Mixture HMM")
    print("=" * 60)
    
    # Parameters
    n_samples = 100000
    np.random.seed(1234)
    
    # State transition probabilities
    a11 = 0.98
    a22 = 0.95
    
    # State 1: Two components
    c1_1 = 0.6  # Weight of component 1
    mu1_1 = np.array([1.0, 1.0])
    mu1_2 = np.array([-3.0, -3.0])
    sigma1_1 = np.eye(2)
    sigma1_2 = np.array([[1.69, 1.0], [1.0, 1.69]])
    
    # State 2: Two components
    c2_1 = 0.7  # Weight of component 1
    mu2_1 = np.array([-1.0, -1.0])
    mu2_2 = np.array([3.0, 3.0])
    sigma2_1 = np.eye(2)
    sigma2_2 = np.array([[1.69, 1.0], [1.0, 1.69]])
    
    # Generate data
    states = np.zeros(n_samples, dtype=int)
    components = np.zeros(n_samples, dtype=int)
    X = np.zeros((n_samples, 2))
    
    # Initial state
    p = (1 - a22) / ((1 - a11) + (1 - a22))
    states[0] = 0 if np.random.rand() < p else 1
    
    for t in range(n_samples):
        if t > 0:
            # State transition
            if states[t-1] == 0:
                states[t] = 0 if np.random.rand() < a11 else 1
            else:
                states[t] = 1 if np.random.rand() < a22 else 0
                
        # Generate observation
        if states[t] == 0:
            # State 1
            if np.random.rand() < c1_1:
                X[t] = np.random.multivariate_normal(mu1_1, sigma1_1)
                components[t] = 0
            else:
                X[t] = np.random.multivariate_normal(mu1_2, sigma1_2)
                components[t] = 1
        else:
            # State 2
            if np.random.rand() < c2_1:
                X[t] = np.random.multivariate_normal(mu2_1, sigma2_1)
                components[t] = 0
            else:
                X[t] = np.random.multivariate_normal(mu2_2, sigma2_2)
                components[t] = 1
                
    # Split into train and test
    train_size = n_samples // 2
    X_train = X[:train_size]
    X_test = X[train_size:]
    states_test = states[train_size:]
    
    # Fit model
    model = GaussianMixtureHMM(
        n_states=2, 
        n_components=2,
        covariance_type='full',
        random_state=42
    )
    model.fit(X_train)
    
    # Predict on test set
    predicted_states = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == states_test)
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        
    print(f"\nModel Summary:")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    
    print(f"\nMixture Weights:")
    print(model.weights_)
    
    print(f"\nComponent Means:")
    for s in range(2):
        print(f"State {s}:")
        for c in range(2):
            print(f"  Component {c}: {model.means_[s, c]}")
            
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training data
    sample_size = 5000
    axes[0, 0].scatter(X_train[:sample_size, 0], X_train[:sample_size, 1],
                      alpha=0.3, s=10)
    axes[0, 0].set_title('Training Data (first 5000 points)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # Plot predicted states on test data
    colors = ['blue', 'red']
    for s in range(2):
        mask = predicted_states[:sample_size] == s
        axes[0, 1].scatter(X_test[mask, 0][:sample_size], 
                         X_test[mask, 1][:sample_size],
                         alpha=0.3, s=10, c=colors[s], label=f'State {s}')
    axes[0, 1].set_title('Test Data with Predicted States')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    
    # Plot transition matrix
    plot_transition_matrix(model.model.transmat_, ax=axes[1, 0])
    
    # Plot state timeline
    plot_states_timeline(predicted_states[:1000], ax=axes[1, 1],
                        title='Predicted States (first 1000 test samples)')
    
    plt.tight_layout()
    plt.show()
    
    return model, X_train, X_test


def example_4_regime_switching_regression():
    """Example 4: Regime-Switching Regression."""
    print("\n" + "=" * 60)
    print("Example 4: Regime-Switching Regression")
    print("=" * 60)
    
    # Generate data
    n_samples = 4000
    np.random.seed(1234)
    
    # Parameters
    x_lb, x_ub = -8, 0
    pi1 = 0.5
    a11 = 0.95
    a22 = 0.95
    
    # State 1: y = 0 + 1*x + N(0, 2.56)
    beta1 = np.array([0.0, 1.0])
    sigma1 = np.sqrt(2.56)
    
    # State 2: y = 4 + 1.5*x + N(0, 4)
    beta2 = np.array([4.0, 1.5])
    sigma2 = np.sqrt(4.0)
    
    # Generate states
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0 if np.random.rand() < pi1 else 1
    
    for t in range(1, n_samples):
        if states[t-1] == 0:
            states[t] = 0 if np.random.rand() < a11 else 1
        else:
            states[t] = 1 if np.random.rand() < a22 else 0
            
    # Generate observations
    X = np.random.uniform(x_lb, x_ub, n_samples)
    y = np.zeros(n_samples)
    
    for t in range(n_samples):
        if states[t] == 0:
            y[t] = beta1[0] + beta1[1] * X[t] + sigma1 * np.random.randn()
        else:
            y[t] = beta2[0] + beta2[1] * X[t] + sigma2 * np.random.randn()
            
    # Fit model
    model = RegimeSwitchingRegression(n_states=2)
    model.fit(X.reshape(-1, 1), y)
    
    # Get results
    predicted_states = model.predict_states(X.reshape(-1, 1), y)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == states)
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        
    print(f"\nModel Summary:")
    print(f"Accuracy: {accuracy:.3f}")
    
    print(f"\nEstimated Coefficients:")
    for s in range(2):
        print(f"State {s}: intercept={model.coeffs_[s, 0]:.3f}, "
              f"slope={model.coeffs_[s, 1]:.3f}, sigma={model.sigmas_[s]:.3f}")
        
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scatter plot with true states
    colors = ['blue', 'red']
    for s in range(2):
        mask = states == s
        axes[0, 0].scatter(X[mask], y[mask], alpha=0.3, s=10, 
                         c=colors[s], label=f'State {s}')
    axes[0, 0].set_title('True States')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    
    # Plot 2: Scatter plot with predicted states
    for s in range(2):
        mask = predicted_states == s
        axes[0, 1].scatter(X[mask], y[mask], alpha=0.3, s=10,
                         c=colors[s], label=f'State {s}')
        
        # Add regression lines
        x_range = np.linspace(x_lb, x_ub, 100)
        y_pred = model.coeffs_[s, 0] + model.coeffs_[s, 1] * x_range
        axes[0, 1].plot(x_range, y_pred, color=colors[s], linewidth=2)
        
    axes[0, 1].set_title('Predicted States with Regression Lines')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    
    # Plot 3: Transition matrix
    plot_transition_matrix(model.model.transmat_, ax=axes[1, 0])
    
    # Plot 4: State timeline
    plot_states_timeline(predicted_states[:500], ax=axes[1, 1],
                        title='Predicted States (first 500 samples)')
    
    plt.tight_layout()
    plt.show()
    
    return model


def example_5_regime_switching_ar():
    """Example 5: Regime-Switching AR model."""
    print("\n" + "=" * 60)
    print("Example 5: Regime-Switching AR Model")
    print("=" * 60)
    
    # Generate data
    n_samples = 1000
    np.random.seed(1234)
    
    # Parameters
    pi1 = 0.5
    a11 = 0.95
    a22 = 0.95
    
    # State 1: AR(1) with phi=0.8
    ar1 = 0.8
    sigma1 = np.sqrt(2.56)
    
    # State 2: AR(1) with phi=-0.7
    ar2 = -0.7
    sigma2 = np.sqrt(4.0)
    
    # Generate states
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0 if np.random.rand() < pi1 else 1
    
    for t in range(1, n_samples):
        if states[t-1] == 0:
            states[t] = 0 if np.random.rand() < a11 else 1
        else:
            states[t] = 1 if np.random.rand() < a22 else 0
            
    # Generate AR series
    y = np.zeros(n_samples)
    y[0] = np.random.randn()  # Initial value
    
    for t in range(1, n_samples):
        if states[t] == 0:
            y[t] = ar1 * y[t-1] + sigma1 * np.random.randn()
        else:
            y[t] = ar2 * y[t-1] + sigma2 * np.random.randn()
            
    # Fit model
    model = RegimeSwitchingAR(n_states=2, ar_order=1, include_constant=False)
    model.fit(y)
    
    # Get results
    predicted_states = model.predict_states(y)
    
    # Calculate accuracy (accounting for lag)
    accuracy = np.mean(predicted_states == states[1:])
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        
    print(f"\nModel Summary:")
    print(f"Accuracy: {accuracy:.3f}")
    
    print(f"\nEstimated AR Parameters:")
    for s in range(2):
        print(f"State {s}: AR(1)={model.ar_params_[s, 0]:.3f}, "
              f"sigma={model.sigmas_[s]:.3f}")
        
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time series
    axes[0, 0].plot(y, alpha=0.7)
    axes[0, 0].set_title('AR Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Y')
    
    # Plot 2: ACF by state
    from statsmodels.graphics.tsaplots import plot_acf
    
    # Separate series by predicted state
    state0_indices = np.where(predicted_states == 0)[0] + 1  # Add 1 for original indices
    state1_indices = np.where(predicted_states == 1)[0] + 1
    
    # Simple ACF plot for the full series
    plot_acf(y, lags=20, ax=axes[0, 1])
    axes[0, 1].set_title('Autocorrelation Function')
    
    # Plot 3: Scatter plot y_t vs y_{t-1} colored by state
    colors = ['blue', 'red']
    for s in range(2):
        mask = predicted_states == s
        axes[1, 0].scatter(y[:-1][mask], y[1:][mask], alpha=0.5, s=20,
                         c=colors[s], label=f'State {s}')
    axes[1, 0].set_xlabel('$y_{t-1}$')
    axes[1, 0].set_ylabel('$y_t$')
    axes[1, 0].set_title('AR Relationship by State')
    axes[1, 0].legend()
    
    # Plot 4: State timeline
    plot_states_timeline(predicted_states, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return model


def example_6_discrete_hmm():
    """Example 6: Discrete HMM for finite observations."""
    print("\n" + "=" * 60)
    print("Example 6: Discrete HMM")
    print("=" * 60)
    
    # Generate data with 2 states and 6 possible symbols
    n_samples = 1000
    np.random.seed(3)
    
    # Parameters
    pi1 = 0.5
    a11 = 0.75
    a22 = 0.75
    
    # Emission probabilities
    # State 1: symbols 0,1,2 equally likely
    # State 2: symbols 3,4,5 equally likely
    emission_probs = np.array([
        [1/3, 1/3, 1/3, 0, 0, 0],
        [0, 0, 0, 1/3, 1/3, 1/3]
    ])
    
    # Generate states
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0 if np.random.rand() < pi1 else 1
    
    for t in range(1, n_samples):
        if states[t-1] == 0:
            states[t] = 0 if np.random.rand() < a11 else 1
        else:
            states[t] = 1 if np.random.rand() < a22 else 0
            
    # Generate observations
    observations = np.zeros(n_samples, dtype=int)
    for t in range(n_samples):
        observations[t] = np.random.choice(6, p=emission_probs[states[t]])
        
    # Fit model
    model = DiscreteHMM(n_states=2, n_symbols=6, random_state=42)
    model.fit(observations.reshape(-1, 1))
    
    # Test if emissions follow uniform distribution within states
    print(f"\nEmission Probabilities:")
    print(model.emissionprob_)
    
    # Chi-square test for uniformity
    from scipy.stats import chisquare
    
    # For state 0, test if first 3 symbols are uniform
    state0_probs = model.emissionprob_[0, :3] / model.emissionprob_[0, :3].sum()
    expected = np.ones(3) / 3
    chi2, p_value = chisquare(state0_probs, expected)
    
    print(f"\nChi-square test for uniformity (State 0, symbols 0-2):")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Observation sequence
    axes[0, 0].plot(observations[:200], 'o-', markersize=3, alpha=0.7)
    axes[0, 0].set_title('Observation Sequence (first 200)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Symbol')
    axes[0, 0].set_ylim(-0.5, 5.5)
    
    # Plot 2: Histogram of observations
    axes[0, 1].hist(observations, bins=np.arange(7) - 0.5, alpha=0.7)
    axes[0, 1].set_title('Distribution of Symbols')
    axes[0, 1].set_xlabel('Symbol')
    axes[0, 1].set_ylabel('Count')
    
    # Plot 3: Emission probabilities
    from src.utils.visualization import plot_discrete_emissions
    plot_discrete_emissions(model.emissionprob_, ax=axes[1, 0])
    
    # Plot 4: Transition matrix
    plot_transition_matrix(model.model.transmat_, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return model


def example_7_poisson_hmm():
    """Example 7: Poisson HMM for count data."""
    print("\n" + "=" * 60)
    print("Example 7: Poisson HMM")
    print("=" * 60)
    
    # Generate count data
    n_samples = 1000
    n_sections = 2
    np.random.seed(1234)
    
    # Parameters
    pi1 = 0.7
    a11 = 0.9
    a22 = 0.8
    lambda1 = 5   # Low count state
    lambda2 = 10  # High count state
    
    data = []
    true_states = []
    
    for sec in range(n_sections):
        # Initial state
        state = 0 if np.random.rand() < pi1 else 1
        
        for t in range(n_samples // n_sections):
            # Generate observation
            if state == 0:
                y = np.random.poisson(lambda1)
            else:
                y = np.random.poisson(lambda2)
                
            data.append({'sec': sec, 't': t, 'y': y})
            true_states.append(state)
            
            # State transition
            if state == 0:
                state = 0 if np.random.rand() < a11 else 1
            else:
                state = 1 if np.random.rand() < a22 else 0
                
    df = pd.DataFrame(data)
    
    # Fit model
    model = PoissonHMM(n_states=2, random_state=42)
    lengths = [n_samples // n_sections] * n_sections
    model.fit(df[['y']].values, lengths=lengths)
    
    # Sort states by lambda
    model.label_switch(sort_by='lambda', ascending=True)
    
    print(f"\nEstimated Parameters:")
    print(f"Lambda values: {model.lambdas_.flatten()}")
    print(f"\nTransition Matrix:")
    print(model.model.transmat_)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time series
    section_data = df[df['sec'] == 0]
    axes[0, 0].plot(section_data['t'], section_data['y'], 'o-', 
                   markersize=3, alpha=0.7)
    axes[0, 0].set_title('Count Time Series (Section 0)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Histogram
    axes[0, 1].hist(df['y'], bins=20, alpha=0.7, density=True)
    
    # Overlay Poisson distributions
    x_range = np.arange(0, df['y'].max() + 1)
    for i, lam in enumerate(model.lambdas_.flatten()):
        pmf = stats.poisson.pmf(x_range, lam)
        axes[0, 1].plot(x_range, pmf, 'o-', label=f'State {i}: λ={lam:.1f}')
    
    axes[0, 1].set_title('Count Distribution with Fitted Poisson')
    axes[0, 1].set_xlabel('Count')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].legend()
    
    # Plot 3: State timeline
    predicted_states = model.predict(df[['y']].values, lengths=lengths)
    plot_states_timeline(predicted_states[:n_samples//n_sections], ax=axes[1, 0],
                        title='Predicted States (Section 0)')
    
    # Plot 4: Transition matrix
    plot_transition_matrix(model.model.transmat_, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return model, df


def main():
    """Run all examples."""
    # Example 1: Basic Gaussian HMM
    model1, df1 = example_1_basic_gaussian_hmm()
    
    # Example 2: Time series clustering
    model2, df2 = example_2_time_series_clustering()
    
    # Example 3: Gaussian Mixture HMM
    model3, X_train, X_test = example_3_gaussian_mixture_hmm()
    
    # Example 4: Regime-switching regression
    model4 = example_4_regime_switching_regression()
    
    # Example 5: Regime-switching AR
    model5 = example_5_regime_switching_ar()
    
    # Example 6: Discrete HMM
    model6 = example_6_discrete_hmm()
    
    # Example 7: Poisson HMM
    model7, df7 = example_7_poisson_hmm()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    return {
        'gaussian': model1,
        'clustering': model2,
        'mixture': model3,
        'regression': model4,
        'ar': model5,
        'discrete': model6,
        'poisson': model7
    }


if __name__ == "__main__":
    models = main()