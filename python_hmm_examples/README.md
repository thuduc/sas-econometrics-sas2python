# Python Hidden Markov Model Examples

This package provides Python implementations of Hidden Markov Models (HMM) corresponding to SAS PROC HMM examples. It's part of the SAS Econometrics to Python migration project.

## Features

### HMM Models Implemented

1. **Gaussian HMM** - For continuous observations with Gaussian emissions
2. **Gaussian Mixture HMM** - For continuous observations with mixture of Gaussians
3. **Discrete/Multinomial HMM** - For discrete/categorical observations
4. **Poisson HMM** - For count data
5. **Regime-Switching Models** - AR and regression models with regime switching

### Examples

1. **Getting Started** (`getting_started.py`) - Basic HMM demonstrations
2. **Market States** (`market_states.py`) - Discovering hidden market regimes
3. **Time Series Clustering** (`time_series_clustering.py`) - Clustering time series by patterns

### Utilities

- **Data Converter** - Convert SAS datasets to CSV format
- **Visualization** - Comprehensive plotting functions for HMM analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd python_hmm_examples

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Basic Gaussian HMM

```python
import numpy as np
from src.models import GaussianHMM

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 2)

# Fit HMM with 2 states
model = GaussianHMM(n_states=2)
model.fit(X)

# Predict hidden states
states = model.predict(X)

# Get model parameters
print(model.results.summary())
```

### Regime-Switching Model

```python
from src.models import RegimeSwitchingAR

# Generate AR time series
y = np.random.randn(500)

# Fit regime-switching AR(1) model
model = RegimeSwitchingAR(n_states=2, ar_order=1)
model.fit(y)

# Predict regimes
regimes = model.predict_states(y)
```

### Run Examples

```python
# Getting started examples
from src.examples import getting_started
results = getting_started.main()

# Market states analysis
from src.examples import market_states
results = market_states.main()

# Time series clustering
from src.examples import time_series_clustering
results = time_series_clustering.main()
```

## Model Details

### Base HMM Class

All models inherit from `BaseHMM` which provides:
- Common fitting interface
- State prediction (Viterbi algorithm)
- Posterior probabilities
- Model evaluation (AIC, BIC)
- Sampling from fitted models
- Stationary distribution calculation

### Model-Specific Features

**Gaussian HMM:**
- Multiple covariance types (spherical, diagonal, full, tied)
- Label switching to handle identifiability
- State distribution extraction

**Discrete HMM:**
- Automatic symbol detection
- Symbol encoding/decoding utilities
- Entropy-based state sorting

**Poisson HMM:**
- Rate parameter constraints
- Support for multiple count features

**Regime-Switching:**
- AR and regression variants
- Switching coefficients and/or variance
- Forecasting capabilities

## Visualization

```python
from src.utils.visualization import plot_hmm_summary

# Create comprehensive summary plot
fig = plot_hmm_summary(model, data)
plt.show()
```

Available visualization functions:
- `plot_states_timeline` - State sequence over time
- `plot_state_probabilities` - Posterior probabilities
- `plot_transition_matrix` - Transition probability heatmap
- `plot_gaussian_emissions` - 2D Gaussian distributions
- `plot_discrete_emissions` - Discrete emission probabilities
- `plot_model_selection` - AIC/BIC for model selection

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models/test_gaussian_hmm.py
```

## Comparison with SAS

This implementation aims to provide similar functionality to SAS PROC HMM:

| SAS Feature | Python Implementation |
|------------|---------------------|
| `MODEL / TYPE=GAUSSIAN` | `GaussianHMM` |
| `MODEL / TYPE=GAUSSIANMIXTURE` | `GaussianMixtureHMM` |
| `MODEL / TYPE=FINITE` | `DiscreteHMM` |
| `MODEL / TYPE=POISSON` | `PoissonHMM` |
| `MODEL / TYPE=REG` | `RegimeSwitchingRegression` |
| `MODEL / TYPE=AR` | `RegimeSwitchingAR` |
| `DECODE` | `model.predict()` |
| `SMOOTH` | `model.predict_proba()` |
| `FILTER` | Available through `hmmlearn` backend |
| `LABELSWITCH` | `model.label_switch()` |

## Requirements

- Python >= 3.8
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- pandas >= 2.0.0
- hmmlearn >= 0.3.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on SAS PROC HMM examples
- Uses `hmmlearn` library for core HMM functionality
- Inspired by econometric applications in finance and time series analysis