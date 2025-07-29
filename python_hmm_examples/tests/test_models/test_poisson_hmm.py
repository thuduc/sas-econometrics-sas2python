"""Tests for Poisson HMM implementation."""

import pytest
import numpy as np
import pandas as pd
from src.models import PoissonHMM


class TestPoissonHMM:
    """Test cases for PoissonHMM."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2-state Poisson data."""
        np.random.seed(42)
        
        # State 0: lambda=2
        # State 1: lambda=10
        lambdas = [2, 10]
        
        n_samples = 500
        states = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        observations = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            observations[i] = np.random.poisson(lambdas[states[i]])
            
        return observations.reshape(-1, 1), states, lambdas
    
    def test_initialization(self):
        """Test model initialization."""
        model = PoissonHMM(n_states=2)
        assert model.n_states == 2
        assert model.n_features == 1
        assert model.min_lambda == 1e-3
        assert not model.fitted
        
    def test_fit_predict(self, simple_data):
        """Test fitting and predicting."""
        X, true_states, true_lambdas = simple_data
        
        model = PoissonHMM(n_states=2, random_state=42)
        model.fit(X)
        
        assert model.fitted
        assert hasattr(model, 'model')
        assert model.model is not None
        
        # Check lambda parameters
        assert model.lambdas_.shape == (2, 1)
        assert np.all(model.lambdas_ > 0)
        
        # Predict states
        predicted_states = model.predict(X)
        assert len(predicted_states) == len(X)
        assert set(predicted_states) == {0, 1}
        
    def test_parameter_estimation(self, simple_data):
        """Test parameter estimation accuracy."""
        X, true_states, true_lambdas = simple_data
        
        model = PoissonHMM(n_states=2, random_state=42, n_iter=200)
        model.fit(X)
        
        # Apply label switching to match true parameters
        model.label_switch(sort_by='lambda', ascending=True)
        
        # Check if estimated lambdas are close to true values
        estimated_lambdas = model.lambdas_.flatten()
        
        # Should be within reasonable range
        assert abs(estimated_lambdas[0] - true_lambdas[0]) < 1.0
        assert abs(estimated_lambdas[1] - true_lambdas[1]) < 2.0
        
    def test_score(self, simple_data):
        """Test log-likelihood computation."""
        X, _, _ = simple_data
        
        model = PoissonHMM(n_states=2, random_state=42)
        model.fit(X)
        
        score = model.score(X)
        assert isinstance(score, float)
        assert score < 0  # Log-likelihood should be negative
        
    def test_state_distribution(self, simple_data):
        """Test getting state distribution parameters."""
        X, _, _ = simple_data
        
        model = PoissonHMM(n_states=2, random_state=42)
        model.fit(X)
        
        for state in range(2):
            dist = model.get_state_distribution(state)
            assert 'lambda' in dist
            assert dist['lambda'].shape == (1,)
            assert dist['lambda'][0] > 0
            
    def test_label_switch(self, simple_data):
        """Test label switching functionality."""
        X, _, _ = simple_data
        
        model = PoissonHMM(n_states=2, random_state=42)
        model.fit(X)
        
        # Get original lambdas
        original_lambdas = model.lambdas_.copy()
        
        # Apply label switching
        new_order = model.label_switch(sort_by='lambda', ascending=True)
        
        # Check that lambdas are sorted
        assert model.lambdas_[0, 0] <= model.lambdas_[1, 0]
        
    def test_with_pandas(self, simple_data):
        """Test with pandas DataFrame input."""
        X, _, _ = simple_data
        df = pd.DataFrame(X, columns=['counts'])
        
        model = PoissonHMM(n_states=2, random_state=42)
        model.fit(df)
        
        predicted = model.predict(df)
        assert len(predicted) == len(df)
        
    def test_multiple_features(self):
        """Test with multiple Poisson features."""
        np.random.seed(42)
        
        # Generate data with 2 features
        n_samples = 300
        states = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        
        # Different lambdas for each feature and state
        lambdas = np.array([[3, 5],    # State 0: lambda1=3, lambda2=5
                           [8, 12]])   # State 1: lambda1=8, lambda2=12
        
        X = np.zeros((n_samples, 2), dtype=int)
        for i in range(n_samples):
            X[i, 0] = np.random.poisson(lambdas[states[i], 0])
            X[i, 1] = np.random.poisson(lambdas[states[i], 1])
            
        model = PoissonHMM(n_states=2, n_features=2, random_state=42)
        model.fit(X)
        
        assert model.fitted
        assert model.lambdas_.shape == (2, 2)
        
    def test_error_handling(self):
        """Test error handling."""
        # Negative counts should raise error
        X_negative = np.array([1, 2, -1, 3]).reshape(-1, 1)
        model = PoissonHMM(n_states=2)
        
        with pytest.raises(ValueError):
            model.fit(X_negative)
            
        # Non-integer counts should give warning
        X_float = np.array([1.5, 2.7, 3.2, 4.8]).reshape(-1, 1)
        model = PoissonHMM(n_states=2)
        
        with pytest.warns(UserWarning):
            model.fit(X_float)
            
    def test_min_lambda_constraint(self):
        """Test minimum lambda constraint."""
        # Generate data with very low counts
        np.random.seed(42)
        X = np.zeros((100, 1), dtype=int)  # All zeros
        
        model = PoissonHMM(n_states=2, min_lambda=0.1, random_state=42)
        model.fit(X)
        
        # All lambdas should be at least min_lambda
        assert np.all(model.lambdas_ >= 0.1)