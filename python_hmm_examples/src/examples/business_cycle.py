"""
Business Cycle Analysis using Hidden Markov Models

This example demonstrates:
- Using HMM to identify recession and expansion periods in economic data
- Fitting a 2-state AR(4) HMM to GNP growth rates
- Comparing HMM-identified recessions with NBER recession dates
- Visualizing filtered and smoothed state probabilities

Based on SAS/ETS HMM Example 3: Analysis of the Business Cycle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from ..models.regime_switching import RegimeSwitchingHMM


def create_gnp_data():
    """Create the GNP dataset from the SAS example"""
    
    # GNP data from 1951q1 to 1984q4
    data = {
        'date': pd.date_range('1951-01-01', '1984-10-01', freq='QS'),
        'gnp': [
            1286.6, 1320.4, 1349.8, 1356.0, 1369.2, 1365.9, 1378.2, 1406.8,
            1431.4, 1444.9, 1438.2, 1426.6, 1406.8, 1401.2, 1418.0, 1438.3,
            1469.6, 1485.7, 1505.5, 1518.7, 1515.7, 1522.6, 1523.7, 1540.6,
            1553.3, 1552.4, 1561.5, 1537.3, 1506.1, 1514.2, 1550.0, 1568.7,
            1606.4, 1634.0, 1629.5, 1634.4, 1671.6, 1666.8, 1668.4, 1654.1,
            1671.3, 1692.1, 1716.3, 1754.9, 1777.9, 1796.4, 1813.1, 1810.1,
            1834.6, 1860.0, 1892.5, 1906.1, 1948.7, 1965.4, 1985.2, 1993.7,
            2036.9, 2066.4, 2099.3, 2147.6, 2190.1, 2195.8, 2218.3, 2229.2,
            2241.8, 2255.2, 2287.7, 2300.6, 2327.3, 2366.9, 2385.3, 2383.0,
            2416.5, 2419.8, 2433.2, 2423.5, 2408.6, 2406.5, 2435.8, 2413.8,
            2478.6, 2478.4, 2491.1, 2491.0, 2545.6, 2595.1, 2622.1, 2671.3,
            2734.0, 2741.0, 2738.3, 2762.8, 2747.4, 2755.2, 2719.3, 2695.4,
            2642.7, 2669.6, 2714.9, 2752.7, 2804.4, 2816.9, 2828.6, 2856.8,
            2896.0, 2942.7, 3001.8, 2994.1, 3020.5, 3115.9, 3142.6, 3181.6,
            3181.7, 3178.7, 3207.4, 3201.3, 3233.4, 3157.0, 3159.1, 3199.2,
            3261.1, 3250.2, 3264.6, 3219.0, 3170.4, 3179.9, 3154.5, 3159.3,
            3190.6, 3259.3, 3303.4, 3357.2, 3449.4, 3492.6, 3510.4, 3515.6
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate growth rate of GNP (percentage change)
    df['dgnp'] = 100 * (np.log(df['gnp']) - np.log(df['gnp'].shift(1)))
    
    # Drop the first row with NaN
    df = df.dropna()
    
    return df


def create_nber_recession_data():
    """Create NBER recession indicator data"""
    
    # NBER recession dates (1 = recession, 0 = expansion)
    # This matches the data in the SAS example
    dates = pd.date_range('1951-01-01', '1984-10-01', freq='QS')
    
    recession_periods = [
        ('1953-07-01', '1954-04-01'),
        ('1957-07-01', '1958-04-01'),
        ('1960-04-01', '1961-01-01'),
        ('1969-10-01', '1970-10-01'),
        ('1973-10-01', '1975-01-01'),
        ('1980-01-01', '1980-07-01'),
        ('1981-07-01', '1982-10-01')
    ]
    
    recession_indicator = np.zeros(len(dates))
    
    for start, end in recession_periods:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        mask = (dates >= start_date) & (dates <= end_date)
        recession_indicator[mask] = 1
    
    df = pd.DataFrame({
        'date': dates,
        'recessionNBER': recession_indicator.astype(int)
    })
    
    # Remove first row to match GNP data
    df = df.iloc[1:].reset_index(drop=True)
    
    return df


def plot_gnp_data(df):
    """Plot GNP and growth rate data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot GNP
    ax1.plot(df['date'], df['gnp'], linewidth=2, color='blue')
    ax1.set_ylabel('GNP (billions)', fontsize=12)
    ax1.set_title('U.S. Gross National Product (1951-1984)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot growth rate
    ax2.plot(df['date'], df['dgnp'], linewidth=2, color='green')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Growth Rate (%)', fontsize=12)
    ax2.set_title('GNP Growth Rate', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def fit_business_cycle_hmm(df, use_initial_params=True):
    """
    Fit a 2-state HMM with AR(4) dynamics to GNP growth data
    
    Parameters:
    -----------
    df : DataFrame
        Data with 'dgnp' column
    use_initial_params : bool
        Whether to use initial parameters from SAS example
    """
    
    # Prepare data with lags for AR(4) model
    y = df['dgnp'].values
    
    # Create lagged features
    lags = 4
    X = []
    y_ar = []
    for i in range(lags, len(y)):
        X.append(y[i-lags:i][::-1])  # Reverse to match AR ordering
        y_ar.append(y[i])
    
    X = np.array(X)
    y_ar = np.array(y_ar)
    
    # Initialize the model
    model = RegimeSwitchingHMM(n_states=2, ar_order=lags)
    
    if use_initial_params:
        # Use initial parameters from SAS example
        model.transition_matrix = np.array([
            [0.9049, 0.0951],
            [0.2450, 0.7550]
        ])
        
        # AR coefficients (same for both states in this example)
        ar_coef = np.array([0.014, -0.058, -0.247, -0.213])
        model.ar_coefficients = np.array([ar_coef, ar_coef])
        
        # Intercepts
        model.intercepts = np.array([1.1643, -0.3577])
        
        # Variances (same for both states initially)
        model.variances = np.array([0.5914, 0.5914])
        
        # Fit the model starting from these initial values
        model.fit(X, y_ar, init_params=False)
    else:
        # Fit from random initialization
        model.fit(X, y_ar)
    
    # Get filtered and smoothed probabilities
    filtered_probs = model.predict_proba(X)
    smoothed_probs = model.smooth(X, y_ar)
    
    # Get most likely state sequence
    states = model.predict(X)
    
    return model, filtered_probs, smoothed_probs, states


def plot_filtered_probabilities(df, filtered_probs, state_idx=1):
    """Plot filtered probability of being in a specific state"""
    
    # Adjust dates to match the filtered probabilities length
    dates = df['date'].values[4:]  # Skip first 4 due to AR(4) lags
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, filtered_probs[:, state_idx], linewidth=2, color='red')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'Filtered Probability of State {state_idx+1}', fontsize=12)
    plt.title('Filtered Probability of Recession State', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add shading for high probability periods
    for i in range(len(filtered_probs)):
        if filtered_probs[i, state_idx] > 0.5:
            plt.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.show()


def plot_state_comparison(df, states, dgnp_values):
    """Plot growth rate with boom/recession bands"""
    
    # Adjust dates
    dates = df['date'].values[4:]
    
    plt.figure(figsize=(12, 6))
    
    # Create recession and boom indicators
    recession = (states == 1).astype(int)
    boom = (states == 0).astype(int)
    
    # Plot bands
    for i in range(len(states)):
        if recession[i]:
            plt.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       ymin=0, ymax=0.4, alpha=0.3, color='red', 
                       label='Recession' if i == 0 else "")
        if boom[i]:
            plt.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       ymin=0.6, ymax=1, alpha=0.3, color='green', 
                       label='Boom' if i == 0 else "")
    
    # Plot growth rate
    plt.plot(dates, dgnp_values, linewidth=2, color='black', label='GNP Growth Rate')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Growth Rate (%)', fontsize=12)
    plt.title('Business Cycle States and GNP Growth Rate', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_with_nber(df, smoothed_probs, nber_df):
    """Compare HMM-identified recessions with NBER dates"""
    
    # Merge data
    dates = df['date'].values[4:]  # Adjust for AR lags
    
    # Create recession indicator from HMM (using 0.5 threshold)
    hmm_recession = (smoothed_probs[:, 1] >= 0.5).astype(int)
    
    # Align NBER data
    nber_aligned = nber_df[nber_df['date'].isin(dates)]['recessionNBER'].values
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot NBER recessions
    for i in range(len(nber_aligned)):
        if nber_aligned[i]:
            ax1.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       alpha=0.3, color='red')
    ax1.plot(df['gnp'].values[4:], linewidth=2, color='blue')
    ax1.set_ylabel('GNP (billions)', fontsize=12)
    ax1.set_title('NBER Recession Dates', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot HMM-identified recessions
    for i in range(len(hmm_recession)):
        if hmm_recession[i]:
            ax2.axvspan(dates[i], dates[min(i+1, len(dates)-1)], 
                       alpha=0.3, color='red')
    ax2.plot(df['gnp'].values[4:], linewidth=2, color='blue')
    ax2.set_ylabel('GNP (billions)', fontsize=12)
    ax2.set_title('HMM-Identified Recessions', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate agreement metrics
    print("\nComparison of HMM vs NBER Recession Classification:")
    print("=" * 50)
    print(f"Total quarters: {len(hmm_recession)}")
    print(f"Agreement: {np.sum(hmm_recession == nber_aligned) / len(hmm_recession) * 100:.1f}%")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(nber_aligned, hmm_recession),
        index=['NBER: Expansion', 'NBER: Recession'],
        columns=['HMM: Expansion', 'HMM: Recession']
    ))


def main():
    """Main function demonstrating business cycle analysis"""
    
    print("Business Cycle Analysis using Hidden Markov Models")
    print("=" * 60)
    
    # Create datasets
    print("\n1. Loading GNP data...")
    df = create_gnp_data()
    nber_df = create_nber_recession_data()
    
    print(f"   - Data period: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Number of quarters: {len(df)}")
    
    # Plot raw data
    print("\n2. Plotting GNP and growth rate...")
    plot_gnp_data(df)
    
    # Fit HMM with initial parameters
    print("\n3. Fitting HMM with initial parameters from Hamilton (1989)...")
    model, filtered_probs, smoothed_probs, states = fit_business_cycle_hmm(df, use_initial_params=True)
    
    print("\nEstimated parameters:")
    print(f"Transition matrix:\n{model.transition_matrix}")
    print(f"\nState means: {model.intercepts}")
    print(f"State variances: {model.variances}")
    print(f"\nAR coefficients:")
    for i in range(2):
        print(f"  State {i+1}: {model.ar_coefficients[i]}")
    
    # Plot filtered probabilities
    print("\n4. Plotting filtered probability of recession...")
    plot_filtered_probabilities(df, filtered_probs, state_idx=1)
    
    # Plot states with growth rate
    print("\n5. Plotting identified business cycle states...")
    plot_state_comparison(df, states, df['dgnp'].values[4:])
    
    # Compare with NBER
    print("\n6. Comparing with NBER recession dates...")
    compare_with_nber(df, smoothed_probs, nber_df)
    
    # Fit model without initial parameters
    print("\n7. Fitting HMM without initial parameters...")
    model2, filtered_probs2, smoothed_probs2, states2 = fit_business_cycle_hmm(df, use_initial_params=False)
    
    print("\nEstimated parameters (no initialization):")
    print(f"Transition matrix:\n{model2.transition_matrix}")
    print(f"\nState means: {model2.intercepts}")
    print(f"State variances: {model2.variances}")
    
    # State persistence
    print("\n8. State persistence analysis:")
    for i in range(2):
        persistence = 1 / (1 - model.transition_matrix[i, i])
        print(f"   - Expected duration in State {i+1}: {persistence:.1f} quarters")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()