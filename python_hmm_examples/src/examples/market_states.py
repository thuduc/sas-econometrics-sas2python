"""
Discovering Hidden Market States - Python implementation of hmmex01.sas.

This example demonstrates using HMM to identify hidden market regimes/states
in financial time series data. The model discovers bull and bear market states
from market returns data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from typing import Tuple, Optional
import warnings

from src.models import RegimeSwitchingAR, GaussianHMM
from src.utils.visualization import (
    plot_states_timeline,
    plot_state_probabilities,
    plot_transition_matrix
)


def load_market_data(
    symbol: str = "^GSPC",  # S&P 500
    start_date: str = "1990-01-01",
    end_date: str = "2023-12-31",
    cutoff_date: Optional[str] = "2000-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load market data and calculate weekly returns.
    
    Parameters
    ----------
    symbol : str
        Yahoo Finance symbol to download
    start_date : str
        Start date for data
    end_date : str
        End date for data
    cutoff_date : str, optional
        Date to split train/test data
        
    Returns
    -------
    df_all : DataFrame
        All data with weekly returns
    df_train : DataFrame
        Training data (before cutoff)
    df_test : DataFrame
        Test data (after cutoff)
    """
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Download data
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using simulated data instead...")
        return create_simulated_market_data(cutoff_date)
    
    # Calculate returns
    data['Return'] = data['Adj Close'].pct_change() * 100
    
    # Calculate weekly returns (every 5 trading days)
    weekly_data = []
    
    # Group by week
    data['Week'] = pd.to_datetime(data.index).to_period('W')
    weekly_returns = data.groupby('Week').agg({
        'Adj Close': 'last',
        'Return': lambda x: ((1 + x/100).prod() - 1) * 100  # Compound returns
    })
    
    # Create DataFrame
    df_all = pd.DataFrame({
        'Date': weekly_returns.index.to_timestamp(),
        'Price': weekly_returns['Adj Close'],
        'Return': weekly_returns['Return']
    }).reset_index(drop=True)
    
    # Remove first row with NaN return
    df_all = df_all.iloc[1:].reset_index(drop=True)
    
    # Split train/test
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        df_train = df_all[df_all['Date'] <= cutoff].copy()
        df_test = df_all[df_all['Date'] > cutoff].copy()
    else:
        # 80/20 split
        n_train = int(len(df_all) * 0.8)
        df_train = df_all.iloc[:n_train].copy()
        df_test = df_all.iloc[n_train:].copy()
    
    print(f"Data loaded: {len(df_all)} weeks total")
    print(f"Training data: {len(df_train)} weeks")
    print(f"Test data: {len(df_test)} weeks")
    
    return df_all, df_train, df_test


def create_simulated_market_data(
    cutoff_date: Optional[str] = "2000-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create simulated market data if real data cannot be downloaded."""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='1990-01-01', end='2023-12-31', freq='W')
    n_samples = len(dates)
    
    # Simulate regime-switching returns
    # Bull market: mean=0.2%, std=2%
    # Bear market: mean=-0.1%, std=3%
    
    states = np.zeros(n_samples, dtype=int)
    returns = np.zeros(n_samples)
    
    # Initial state
    states[0] = 0  # Start in bull market
    
    # State transition probabilities
    p_bull_to_bull = 0.95
    p_bear_to_bear = 0.90
    
    for t in range(1, n_samples):
        # State transition
        if states[t-1] == 0:  # Bull
            states[t] = 0 if np.random.rand() < p_bull_to_bull else 1
        else:  # Bear
            states[t] = 1 if np.random.rand() < p_bear_to_bear else 0
            
        # Generate return based on state
        if states[t] == 0:  # Bull
            returns[t] = np.random.normal(0.2, 2.0)
        else:  # Bear
            returns[t] = np.random.normal(-0.1, 3.0)
    
    # Calculate prices
    prices = 100 * np.exp(np.cumsum(returns / 100))
    
    # Create DataFrame
    df_all = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Return': returns
    })
    
    # Split train/test
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        df_train = df_all[df_all['Date'] <= cutoff].copy()
        df_test = df_all[df_all['Date'] > cutoff].copy()
    else:
        n_train = int(len(df_all) * 0.8)
        df_train = df_all.iloc[:n_train].copy()
        df_test = df_all.iloc[n_train:].copy()
    
    return df_all, df_train, df_test


def analyze_simple_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_states: int = 2
) -> Tuple[RegimeSwitchingAR, pd.DataFrame]:
    """
    Analyze market states using a simple AR(0) model (no lags).
    
    This is equivalent to a Gaussian HMM with different means and variances
    for each state.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS WITH SIMPLE MODEL (AR(0))")
    print("=" * 60)
    
    # Fit model on training data
    model = RegimeSwitchingAR(
        n_states=n_states,
        ar_order=0,
        switching_ar=False,  # Only switching variance
        switching_variance=True,
        include_constant=True,
        random_state=42
    )
    
    model.fit(df_train['Return'].values)
    
    # Get state probabilities
    train_states = model.predict_states(df_train['Return'].values)
    train_probs = model.predict_proba(
        model._compute_residuals(
            model._create_ar_matrix(df_train['Return'].values),
            df_train['Return'].values[model.ar_order:]
        ).reshape(-1, 1)
    )
    
    # Calculate state statistics
    print("\nState Statistics (Training Data):")
    for state in range(n_states):
        mask = train_states == state
        returns_in_state = df_train['Return'].values[model.ar_order:][mask]
        
        mean_return = np.mean(returns_in_state)
        std_return = np.std(returns_in_state)
        prob = np.mean(mask)
        
        print(f"\nState {state}:")
        print(f"  Probability: {prob:.3f}")
        print(f"  Mean return: {mean_return:.3f}%")
        print(f"  Std deviation: {std_return:.3f}%")
        print(f"  Sharpe ratio (annualized): {mean_return/std_return * np.sqrt(52):.3f}")
    
    # Transition matrix
    print("\nTransition Matrix:")
    print(model.model.transmat_)
    
    # Create results DataFrame
    results_df = df_train.copy()
    results_df['State'] = np.nan
    results_df.iloc[model.ar_order:, results_df.columns.get_loc('State')] = train_states
    
    # Add probabilities
    for i in range(n_states):
        results_df[f'Prob_State{i}'] = np.nan
        results_df.iloc[model.ar_order:, results_df.columns.get_loc(f'Prob_State{i}')] = train_probs[:, i]
    
    return model, results_df


def analyze_advanced_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> dict:
    """Analyze market states using various model specifications."""
    print("\n" + "=" * 60)
    print("ANALYSIS WITH ADVANCED MODELS")
    print("=" * 60)
    
    results = {}
    
    # Model 1: 2-state AR(1) with switching coefficients
    print("\nModel 1: 2-state AR(1) with regime-switching coefficients")
    model1 = RegimeSwitchingAR(
        n_states=2,
        ar_order=1,
        switching_ar=True,
        switching_variance=True,
        include_constant=True,
        random_state=42
    )
    model1.fit(df_train['Return'].values)
    results['ar1_2state'] = model1
    
    # Model 2: 3-state AR(0) - Bull, Bear, Crisis
    print("\nModel 2: 3-state AR(0) - Bull/Normal/Bear")
    model2 = RegimeSwitchingAR(
        n_states=3,
        ar_order=0,
        switching_ar=False,
        switching_variance=True,
        include_constant=True,
        random_state=42
    )
    model2.fit(df_train['Return'].values)
    results['ar0_3state'] = model2
    
    # Model 3: 2-state AR(2)
    print("\nModel 3: 2-state AR(2) with regime-switching")
    model3 = RegimeSwitchingAR(
        n_states=2,
        ar_order=2,
        switching_ar=True,
        switching_variance=True,
        include_constant=True,
        random_state=42
    )
    model3.fit(df_train['Return'].values)
    results['ar2_2state'] = model3
    
    # Compare models using information criteria
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'Log-Likelihood':<15} {'AIC':<10} {'BIC':<10}")
    print("-" * 55)
    
    for name, model in results.items():
        ll = model.score(
            model._compute_residuals(
                model._create_ar_matrix(df_train['Return'].values),
                df_train['Return'].values[model.ar_order:]
            ).reshape(-1, 1)
        )
        n_params = model._count_parameters() + model._count_model_parameters()
        n_obs = len(df_train) - model.ar_order
        
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_obs) - 2 * ll
        
        print(f"{name:<20} {ll:<15.2f} {aic:<10.2f} {bic:<10.2f}")
    
    return results


def plot_market_analysis(
    df: pd.DataFrame,
    model: RegimeSwitchingAR,
    title_prefix: str = ""
) -> plt.Figure:
    """Create comprehensive visualization of market state analysis."""
    # Predict states
    states = model.predict_states(df['Return'].values)
    
    # Adjust dataframe indices for AR lag
    df_adjusted = df.iloc[model.ar_order:].copy()
    df_adjusted['State'] = states
    
    # Identify state characteristics
    state_stats = []
    for state in range(model.n_states):
        mask = states == state
        returns = df_adjusted.loc[mask, 'Return']
        state_stats.append({
            'state': state,
            'mean': returns.mean(),
            'std': returns.std(),
            'count': len(returns)
        })
    
    # Sort states by mean return
    state_stats = sorted(state_stats, key=lambda x: x['mean'], reverse=True)
    state_labels = {s['state']: f"State {i}" for i, s in enumerate(state_stats)}
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"{title_prefix}Hidden Market States Analysis", fontsize=16)
    
    # Plot 1: Price and Returns
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Plot price on left axis
    ax1.plot(df_adjusted['Date'], df_adjusted['Price'], 'b-', alpha=0.7, label='Price')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot returns on right axis
    ax1_twin.bar(df_adjusted['Date'], df_adjusted['Return'], 
                 alpha=0.3, color='gray', width=5, label='Weekly Return')
    ax1_twin.set_ylabel('Weekly Return (%)', color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    ax1_twin.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax1.set_title('Market Price and Returns')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: States timeline
    ax2 = axes[1]
    
    # Color states based on characteristics
    colors = ['green', 'yellow', 'red']  # Bull, Normal, Bear
    for i, (state_num, label) in enumerate(state_labels.items()):
        mask = df_adjusted['State'] == state_num
        ax2.fill_between(df_adjusted.loc[mask, 'Date'], 0, 1,
                        alpha=0.7, color=colors[i], label=label)
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('State')
    ax2.set_title('Market Regimes Over Time')
    ax2.legend(loc='upper right')
    ax2.set_yticks([])
    
    # Plot 3: State probabilities (if available)
    ax3 = axes[2]
    
    prob_cols = [col for col in df.columns if col.startswith('Prob_State')]
    if prob_cols:
        for i, col in enumerate(prob_cols):
            ax3.plot(df_adjusted['Date'], df.iloc[model.ar_order:][col],
                    label=f'P(State {i})', alpha=0.8)
        ax3.set_ylabel('Probability')
        ax3.set_title('Smoothed State Probabilities')
        ax3.legend()
        ax3.set_ylim(0, 1)
    else:
        # If no probabilities, show state distribution
        state_counts = df_adjusted['State'].value_counts().sort_index()
        ax3.bar(state_counts.index, state_counts.values)
        ax3.set_xlabel('State')
        ax3.set_ylabel('Count')
        ax3.set_title('State Distribution')
    
    # Plot 4: Returns by state
    ax4 = axes[3]
    
    # Prepare data for box plot
    returns_by_state = []
    labels = []
    for state_num, label in state_labels.items():
        mask = df_adjusted['State'] == state_num
        returns_by_state.append(df_adjusted.loc[mask, 'Return'].values)
        
        # Add statistics to label
        mean = state_stats[state_num]['mean']
        std = state_stats[state_num]['std']
        labels.append(f"{label}\nμ={mean:.2f}%\nσ={std:.2f}%")
    
    box_plot = ax4.boxplot(returns_by_state, labels=labels, patch_artist=True)
    
    # Color the boxes
    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_ylabel('Weekly Return (%)')
    ax4.set_title('Return Distribution by Market Regime')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def evaluate_forecasting(
    model: RegimeSwitchingAR,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_ahead: int = 4
) -> pd.DataFrame:
    """Evaluate out-of-sample forecasting performance."""
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 60)
    
    # Make rolling forecasts
    test_returns = df_test['Return'].values
    n_test = len(test_returns)
    
    forecasts = []
    actual = []
    
    # Use expanding window for forecasting
    train_returns = df_train['Return'].values
    
    for i in range(model.ar_order, n_test - n_ahead):
        # Combine train data with test data up to current point
        all_data = np.concatenate([train_returns, test_returns[:i]])
        
        # Make forecast
        forecast, forecast_std = model.forecast(all_data, steps=n_ahead, return_std=True)
        
        forecasts.append(forecast[0])  # One-step ahead forecast
        actual.append(test_returns[i])
    
    forecasts = np.array(forecasts)
    actual = np.array(actual)
    
    # Calculate metrics
    mse = np.mean((forecasts - actual) ** 2)
    mae = np.mean(np.abs(forecasts - actual))
    
    # Direction accuracy
    direction_correct = np.mean((forecasts > 0) == (actual > 0))
    
    print(f"\nForecast Evaluation ({n_ahead}-step ahead):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"Direction Accuracy: {direction_correct:.3f}")
    
    # Compare with naive forecast (last value)
    naive_forecast = test_returns[model.ar_order-1:n_test-n_ahead-1]
    naive_mse = np.mean((naive_forecast - actual) ** 2)
    
    print(f"\nNaive (last value) MSE: {naive_mse:.4f}")
    print(f"Relative improvement: {(1 - mse/naive_mse)*100:.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': df_test['Date'].iloc[model.ar_order:n_test-n_ahead],
        'Actual': actual,
        'Forecast': forecasts,
        'Error': actual - forecasts,
        'AbsError': np.abs(actual - forecasts)
    })
    
    return results_df


def main():
    """Run the complete market states analysis."""
    # Load data
    df_all, df_train, df_test = load_market_data()
    
    # Simple model analysis
    simple_model, results_df = analyze_simple_model(df_train, df_test)
    
    # Plot simple model results
    fig1 = plot_market_analysis(results_df, simple_model, "Simple Model: ")
    plt.show()
    
    # Advanced models
    advanced_models = analyze_advanced_models(df_train, df_test)
    
    # Select best model (typically lowest BIC)
    best_model = advanced_models['ar1_2state']  # You can change this based on criteria
    
    # Plot best model on full dataset
    fig2 = plot_market_analysis(df_all, best_model, "Best Model: ")
    plt.show()
    
    # Evaluate forecasting
    forecast_results = evaluate_forecasting(best_model, df_train, df_test)
    
    # Plot forecast results
    fig3, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_results['Date'], forecast_results['Actual'], 
            'b-', label='Actual', alpha=0.7)
    ax.plot(forecast_results['Date'], forecast_results['Forecast'], 
            'r--', label='Forecast', alpha=0.7)
    ax.fill_between(forecast_results['Date'], 
                    forecast_results['Forecast'] - 2*forecast_results['AbsError'].rolling(20).mean(),
                    forecast_results['Forecast'] + 2*forecast_results['AbsError'].rolling(20).mean(),
                    alpha=0.2, color='red', label='Uncertainty')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.set_title('Out-of-Sample Forecast Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Market States Analysis Complete!")
    print("=" * 60)
    
    return {
        'simple_model': simple_model,
        'advanced_models': advanced_models,
        'data': {
            'all': df_all,
            'train': df_train,
            'test': df_test
        },
        'forecast_results': forecast_results
    }


if __name__ == "__main__":
    results = main()