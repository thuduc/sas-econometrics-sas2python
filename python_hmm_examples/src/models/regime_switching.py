"""Regime-Switching model implementations (AR and Regression)."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List, Tuple
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
import warnings

from .gaussian_hmm import GaussianHMM


class RegimeSwitchingAR(GaussianHMM):
    """
    Regime-Switching Autoregressive (AR) Hidden Markov Model.
    
    This model combines HMM with autoregressive dynamics, where the AR
    coefficients and variance can switch between different regimes (states).
    
    Parameters
    ----------
    n_states : int
        Number of hidden states (regimes)
    ar_order : int
        Order of the autoregressive model
    switching_ar : bool, default=True
        Whether AR coefficients switch between regimes
    switching_variance : bool, default=True
        Whether variance switches between regimes
    include_constant : bool, default=True
        Whether to include a constant term in the AR model
    **kwargs : additional keyword arguments
        Passed to the GaussianHMM constructor
        
    Attributes
    ----------
    ar_params_ : array, shape (n_states, ar_order + include_constant)
        AR parameters for each state
    sigmas_ : array, shape (n_states,)
        Standard deviations for each state
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import RegimeSwitchingAR
    >>> 
    >>> # Generate synthetic AR data with regime switching
    >>> np.random.seed(42)
    >>> n_samples = 1000
    >>> 
    >>> # Define two regimes
    >>> ar_params = [
    ...     [0.5, 0.8],    # Regime 1: constant=0.5, AR(1)=0.8
    ...     [2.0, -0.7]    # Regime 2: constant=2.0, AR(1)=-0.7
    ... ]
    >>> sigmas = [0.5, 1.0]
    >>> 
    >>> # Generate data (simplified)
    >>> y = np.zeros(n_samples)
    >>> states = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    >>> 
    >>> for t in range(1, n_samples):
    ...     state = states[t]
    ...     y[t] = (ar_params[state][0] + 
    ...             ar_params[state][1] * y[t-1] + 
    ...             sigmas[state] * np.random.randn())
    >>> 
    >>> # Fit model
    >>> model = RegimeSwitchingAR(n_states=2, ar_order=1)
    >>> model.fit(y)
    >>> 
    >>> # Predict regimes
    >>> predicted_states = model.predict(y)
    """
    
    def __init__(
        self,
        n_states: int,
        ar_order: int,
        switching_ar: bool = True,
        switching_variance: bool = True,
        include_constant: bool = True,
        **kwargs
    ):
        """Initialize Regime-Switching AR model."""
        super().__init__(n_states=n_states, n_features=1, **kwargs)
        
        self.ar_order = ar_order
        self.switching_ar = switching_ar
        self.switching_variance = switching_variance
        self.include_constant = include_constant
        
        # Model-specific attributes
        self.ar_params_ = None
        self.sigmas_ = None
        
    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        lengths: Optional[List[int]] = None,
        **kwargs
    ) -> 'RegimeSwitchingAR':
        """
        Fit the Regime-Switching AR model.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Time series data
        lengths : list of int, optional
            Lengths of the individual sequences in y
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : RegimeSwitchingAR
            The fitted model
        """
        # Convert to numpy array
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y).flatten()
        
        # Create lagged data matrix
        X = self._create_ar_matrix(y)
        
        # Initial parameter estimation using k-means on residuals
        if self.ar_params_ is None:
            self._initialize_parameters(y, X)
            
        # Create observations for HMM (residuals)
        residuals = self._compute_residuals(y, X)
        
        # Fit HMM on residuals
        super().fit(residuals.reshape(-1, 1), lengths=lengths, **kwargs)
        
        # Re-estimate AR parameters given states
        self._reestimate_ar_params(y, X)
        
        return self
    
    def _create_ar_matrix(self, y: np.ndarray) -> np.ndarray:
        """Create design matrix for AR model."""
        n = len(y)
        n_vars = self.ar_order + (1 if self.include_constant else 0)
        X = np.zeros((n - self.ar_order, n_vars))
        
        # Add constant if included
        col = 0
        if self.include_constant:
            X[:, col] = 1
            col += 1
            
        # Add lagged values
        for lag in range(1, self.ar_order + 1):
            X[:, col] = y[self.ar_order - lag:-lag]
            col += 1
            
        return X
    
    def _initialize_parameters(self, y: np.ndarray, X: np.ndarray):
        """Initialize AR parameters using simple estimation."""
        y_trim = y[self.ar_order:]
        
        # Fit single AR model
        ols = OLS(y_trim, X)
        results = ols.fit()
        
        # Initialize parameters for all states
        if self.switching_ar:
            # Add some variation to parameters
            self.ar_params_ = np.zeros((self.n_states, X.shape[1]))
            for i in range(self.n_states):
                noise = np.random.randn(X.shape[1]) * 0.1
                self.ar_params_[i] = results.params + noise
        else:
            # Same AR parameters for all states
            self.ar_params_ = np.tile(results.params, (self.n_states, 1))
            
        if self.switching_variance:
            # Different variances for states
            base_sigma = np.std(results.resid)
            self.sigmas_ = base_sigma * np.linspace(0.5, 1.5, self.n_states)
        else:
            # Same variance for all states
            self.sigmas_ = np.full(self.n_states, np.std(results.resid))
            
        # Set initial parameters for Gaussian HMM
        self.model.means_ = np.zeros((self.n_states, 1))
        self.model.covars_ = self.sigmas_[:, np.newaxis] ** 2
        
    def _compute_residuals(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute residuals for current parameters."""
        y_trim = y[self.ar_order:]
        
        # Get state sequence using current parameters
        if self.fitted and hasattr(self.model, 'predict'):
            # Use current model to predict states
            dummy_residuals = np.zeros((len(y_trim), 1))
            states = self.model.predict(dummy_residuals)
        else:
            # Initial random state assignment
            states = np.random.choice(self.n_states, size=len(y_trim))
            
        # Compute residuals based on state-specific parameters
        residuals = np.zeros(len(y_trim))
        for state in range(self.n_states):
            mask = states == state
            if np.any(mask):
                y_pred = X[mask] @ self.ar_params_[state]
                residuals[mask] = y_trim[mask] - y_pred
                
        return residuals
    
    def _reestimate_ar_params(self, y: np.ndarray, X: np.ndarray):
        """Re-estimate AR parameters given state sequence."""
        y_trim = y[self.ar_order:]
        
        # Get state sequence from fitted HMM
        residuals = self._compute_residuals(y, X)
        states = self.predict(residuals.reshape(-1, 1))
        
        # Re-estimate parameters for each state
        for state in range(self.n_states):
            mask = states == state
            if np.sum(mask) > X.shape[1]:  # Ensure enough observations
                if self.switching_ar or state == 0:
                    # Estimate AR parameters
                    X_state = X[mask]
                    y_state = y_trim[mask]
                    ols = OLS(y_state, X_state)
                    results = ols.fit()
                    self.ar_params_[state] = results.params
                    
                    if self.switching_variance or state == 0:
                        self.sigmas_[state] = np.std(results.resid)
                        
        # Update Gaussian HMM parameters
        self.model.means_ = np.zeros((self.n_states, 1))
        self.model.covars_ = self.sigmas_[:, np.newaxis] ** 2
        
    def predict_states(
        self,
        y: Union[np.ndarray, pd.Series],
        lengths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Predict the hidden states for a time series.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Time series data
        lengths : list of int, optional
            Lengths of the individual sequences
            
        Returns
        -------
        states : array, shape (n_samples - ar_order,)
            Predicted states
        """
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y).flatten()
        
        X = self._create_ar_matrix(y)
        residuals = self._compute_residuals(y, X)
        
        return self.predict(residuals.reshape(-1, 1), lengths=lengths)
    
    def forecast(
        self,
        y: Union[np.ndarray, pd.Series],
        steps: int = 1,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forecast future values.
        
        Parameters
        ----------
        y : array-like
            Historical time series data
        steps : int, default=1
            Number of steps to forecast
        return_std : bool, default=False
            Whether to return standard deviations
            
        Returns
        -------
        forecast : array, shape (steps,)
            Forecasted values
        std : array, shape (steps,), optional
            Standard deviations of forecasts
        """
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y).flatten()
        
        # Get current state probabilities
        X = self._create_ar_matrix(y)
        residuals = self._compute_residuals(y, X)
        state_probs = self.predict_proba(residuals.reshape(-1, 1))[-1]
        
        # Forecast
        forecasts = np.zeros(steps)
        stds = np.zeros(steps)
        y_extended = np.concatenate([y, np.zeros(steps)])
        
        for h in range(steps):
            # Create feature vector for forecast
            x_h = []
            if self.include_constant:
                x_h.append(1)
            for lag in range(1, self.ar_order + 1):
                x_h.append(y_extended[len(y) + h - lag])
            x_h = np.array(x_h)
            
            # Compute forecast as weighted average across states
            forecast_h = 0
            var_h = 0
            for state in range(self.n_states):
                mean_h = x_h @ self.ar_params_[state]
                forecast_h += state_probs[state] * mean_h
                var_h += state_probs[state] * (self.sigmas_[state]**2 + mean_h**2)
                
            var_h -= forecast_h**2
            
            forecasts[h] = forecast_h
            stds[h] = np.sqrt(var_h)
            y_extended[len(y) + h] = forecast_h
            
            # Update state probabilities (simplified - assumes persistence)
            state_probs = self.model.transmat_.T @ state_probs
            
        if return_std:
            return forecasts, stds
        return forecasts


class RegimeSwitchingRegression(GaussianHMM):
    """
    Regime-Switching Regression Hidden Markov Model.
    
    This model combines HMM with linear regression, where the regression
    coefficients and variance can switch between different regimes (states).
    
    Parameters
    ----------
    n_states : int
        Number of hidden states (regimes)
    switching_coeffs : bool, default=True
        Whether regression coefficients switch between regimes
    switching_variance : bool, default=True
        Whether variance switches between regimes
    **kwargs : additional keyword arguments
        Passed to the GaussianHMM constructor
        
    Attributes
    ----------
    coeffs_ : array, shape (n_states, n_features)
        Regression coefficients for each state
    sigmas_ : array, shape (n_states,)
        Standard deviations for each state
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import RegimeSwitchingRegression
    >>> 
    >>> # Generate synthetic regression data with regime switching
    >>> np.random.seed(42)
    >>> n_samples = 1000
    >>> X = np.random.randn(n_samples, 2)
    >>> 
    >>> # Define two regimes with different coefficients
    >>> coeffs = [
    ...     [1.0, 2.0, -0.5],   # Regime 1: intercept=1, beta1=2, beta2=-0.5
    ...     [0.0, -1.0, 1.5]    # Regime 2: intercept=0, beta1=-1, beta2=1.5
    ... ]
    >>> sigmas = [0.5, 1.0]
    >>> 
    >>> # Generate data
    >>> states = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    >>> y = np.zeros(n_samples)
    >>> X_with_const = np.column_stack([np.ones(n_samples), X])
    >>> 
    >>> for i in range(n_samples):
    ...     state = states[i]
    ...     y[i] = (X_with_const[i] @ coeffs[state] + 
    ...             sigmas[state] * np.random.randn())
    >>> 
    >>> # Fit model
    >>> model = RegimeSwitchingRegression(n_states=2)
    >>> model.fit(X, y)
    >>> 
    >>> # Predict regimes
    >>> predicted_states = model.predict_states(X, y)
    """
    
    def __init__(
        self,
        n_states: int,
        switching_coeffs: bool = True,
        switching_variance: bool = True,
        **kwargs
    ):
        """Initialize Regime-Switching Regression model."""
        super().__init__(n_states=n_states, n_features=1, **kwargs)
        
        self.switching_coeffs = switching_coeffs
        self.switching_variance = switching_variance
        
        # Model-specific attributes
        self.coeffs_ = None
        self.sigmas_ = None
        self.n_predictors_ = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        lengths: Optional[List[int]] = None,
        include_constant: bool = True,
        **kwargs
    ) -> 'RegimeSwitchingRegression':
        """
        Fit the Regime-Switching Regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target values
        lengths : list of int, optional
            Lengths of the individual sequences
        include_constant : bool, default=True
            Whether to include an intercept term
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : RegimeSwitchingRegression
            The fitted model
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Add constant if requested
        if include_constant:
            X = np.column_stack([np.ones(len(X)), X])
            
        self.n_predictors_ = X.shape[1]
        
        # Initial parameter estimation
        if self.coeffs_ is None:
            self._initialize_parameters(X, y)
            
        # Create observations for HMM (residuals)
        residuals = self._compute_residuals(X, y)
        
        # Fit HMM on residuals
        super().fit(residuals.reshape(-1, 1), lengths=lengths, **kwargs)
        
        # Re-estimate regression parameters given states
        self._reestimate_params(X, y)
        
        return self
    
    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray):
        """Initialize regression parameters."""
        # Fit single regression model
        ols = OLS(y, X)
        results = ols.fit()
        
        # Initialize parameters for all states
        if self.switching_coeffs:
            # Add some variation to coefficients
            self.coeffs_ = np.zeros((self.n_states, X.shape[1]))
            for i in range(self.n_states):
                noise = np.random.randn(X.shape[1]) * 0.1
                self.coeffs_[i] = results.params + noise
        else:
            # Same coefficients for all states
            self.coeffs_ = np.tile(results.params, (self.n_states, 1))
            
        if self.switching_variance:
            # Different variances for states
            base_sigma = np.std(results.resid)
            self.sigmas_ = base_sigma * np.linspace(0.5, 1.5, self.n_states)
        else:
            # Same variance for all states
            self.sigmas_ = np.full(self.n_states, np.std(results.resid))
            
        # Set initial parameters for Gaussian HMM
        self.model.means_ = np.zeros((self.n_states, 1))
        self.model.covars_ = self.sigmas_[:, np.newaxis] ** 2
        
    def _compute_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals for current parameters."""
        # Get state sequence using current parameters
        if self.fitted and hasattr(self.model, 'predict'):
            # Use current model to predict states
            dummy_residuals = np.zeros((len(y), 1))
            states = self.model.predict(dummy_residuals)
        else:
            # Initial random state assignment
            states = np.random.choice(self.n_states, size=len(y))
            
        # Compute residuals based on state-specific parameters
        residuals = np.zeros(len(y))
        for state in range(self.n_states):
            mask = states == state
            if np.any(mask):
                y_pred = X[mask] @ self.coeffs_[state]
                residuals[mask] = y[mask] - y_pred
                
        return residuals
    
    def _reestimate_params(self, X: np.ndarray, y: np.ndarray):
        """Re-estimate regression parameters given state sequence."""
        # Get state sequence from fitted HMM
        residuals = self._compute_residuals(X, y)
        states = self.predict(residuals.reshape(-1, 1))
        
        # Re-estimate parameters for each state
        for state in range(self.n_states):
            mask = states == state
            if np.sum(mask) > X.shape[1]:  # Ensure enough observations
                if self.switching_coeffs or state == 0:
                    # Estimate regression coefficients
                    X_state = X[mask]
                    y_state = y[mask]
                    ols = OLS(y_state, X_state)
                    results = ols.fit()
                    self.coeffs_[state] = results.params
                    
                    if self.switching_variance or state == 0:
                        self.sigmas_[state] = np.std(results.resid)
                        
        # Update Gaussian HMM parameters
        self.model.means_ = np.zeros((self.n_states, 1))
        self.model.covars_ = self.sigmas_[:, np.newaxis] ** 2
        
    def predict_states(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        lengths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Predict the hidden states.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target values
        lengths : list of int, optional
            Lengths of the individual sequences
            
        Returns
        -------
        states : array, shape (n_samples,)
            Predicted states
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Add constant if it was used in fitting
        if X.shape[1] < self.n_predictors_:
            X = np.column_stack([np.ones(len(X)), X])
            
        residuals = self._compute_residuals(X, y)
        return self.predict(residuals.reshape(-1, 1), lengths=lengths)
    
    def predict_y(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        states: Optional[np.ndarray] = None,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict target values given features and optionally states.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        states : array-like, shape (n_samples,), optional
            Known states. If None, uses predicted states
        return_std : bool, default=False
            Whether to return standard deviations
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
        std : array, shape (n_samples,), optional
            Standard deviations of predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Add constant if it was used in fitting
        if X.shape[1] < self.n_predictors_:
            X = np.column_stack([np.ones(len(X)), X])
            
        if states is None:
            # Predict states using a dummy y
            dummy_y = np.zeros(len(X))
            states = self.predict_states(X, dummy_y)
            
        # Compute predictions
        y_pred = np.zeros(len(X))
        stds = np.zeros(len(X))
        
        for state in range(self.n_states):
            mask = states == state
            if np.any(mask):
                y_pred[mask] = X[mask] @ self.coeffs_[state]
                stds[mask] = self.sigmas_[state]
                
        if return_std:
            return y_pred, stds
        return y_pred