"""Tests for Gaussian HMM implementation."""

import pytest
import numpy as np
import pandas as pd
from src.models import GaussianHMM


class TestGaussianHMM:
    """Test cases for GaussianHMM."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2-state Gaussian data."""
        np.random.seed(42)
        
        # State 0: mean=[0, 0], cov=I
        # State 1: mean=[3, 3], cov=I
        n_samples = 500
        states = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        X = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if states[i] == 0:
                X[i] = np.random.multivariate_normal([0, 0], np.eye(2))
            else:
                X[i] = np.random.multivariate_normal([3, 3], np.eye(2))
                
        return X, states
    
    def test_initialization(self):
        """Test model initialization."""
        model = GaussianHMM(n_states=2)
        assert model.n_states == 2
        assert model.covariance_type == 'diag'
        assert not model.fitted
        
    def test_fit_predict(self, simple_data):
        """Test fitting and predicting."""
        X, true_states = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        assert model.fitted
        assert hasattr(model, 'model')
        assert model.model is not None
        
        # Predict states
        predicted_states = model.predict(X)
        assert len(predicted_states) == len(X)
        assert set(predicted_states) == {0, 1}
        
    def test_score(self, simple_data):
        """Test log-likelihood computation."""
        X, _ = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        score = model.score(X)
        assert isinstance(score, float)
        assert score < 0  # Log-likelihood should be negative
        
    def test_predict_proba(self, simple_data):
        """Test posterior probability computation."""
        X, _ = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1)  # Probabilities sum to 1
        assert np.all(probs >= 0) and np.all(probs <= 1)
        
    def test_sample(self):
        """Test sampling from the model."""
        model = GaussianHMM(n_states=2, random_state=42)
        
        # Initialize with known parameters
        model.model = model._create_model()
        model.model.startprob_ = np.array([0.6, 0.4])
        model.model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        model.model.means_ = np.array([[0, 0], [3, 3]])
        model.model.covars_ = np.array([[1, 1], [1, 1]])
        model.fitted = True
        
        X_sampled, states_sampled = model.sample(n_samples=100)
        
        assert X_sampled.shape == (100, 2)
        assert states_sampled.shape == (100,)
        assert set(states_sampled) <= {0, 1}
        
    def test_stationary_distribution(self, simple_data):
        """Test stationary distribution calculation."""
        X, _ = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        stationary = model.get_stationary_distribution()
        assert len(stationary) == 2
        assert np.allclose(stationary.sum(), 1)
        assert np.all(stationary >= 0)
        
    def test_label_switch(self, simple_data):
        """Test label switching functionality."""
        X, _ = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        # Get original means
        original_means = model.means_.copy()
        
        # Apply label switching
        new_order = model.label_switch(sort_by='means', ascending=True)
        
        # Check that means are sorted
        assert model.means_[0, 0] <= model.means_[1, 0]
        
    def test_results_object(self, simple_data):
        """Test HMMResults object."""
        X, _ = simple_data
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X)
        
        results = model.results
        assert hasattr(results, 'log_likelihood')
        assert hasattr(results, 'aic')
        assert hasattr(results, 'bic')
        assert hasattr(results, 'transition_matrix')
        assert hasattr(results, 'initial_probabilities')
        
        # Check summary method
        summary = results.summary()
        assert isinstance(summary, str)
        assert 'Log-likelihood' in summary
        
    def test_covariance_types(self, simple_data):
        """Test different covariance types."""
        X, _ = simple_data
        
        for cov_type in ['spherical', 'diag', 'full', 'tied']:
            model = GaussianHMM(n_states=2, covariance_type=cov_type, random_state=42)
            model.fit(X)
            
            assert model.fitted
            assert model.covars_ is not None
            
    def test_with_pandas(self, simple_data):
        """Test with pandas DataFrame input."""
        X, _ = simple_data
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(df)
        
        predicted = model.predict(df)
        assert len(predicted) == len(df)
        
    def test_multiple_sequences(self):
        """Test with multiple sequences."""
        np.random.seed(42)
        
        # Generate 3 sequences of different lengths
        seq1 = np.random.randn(50, 2)
        seq2 = np.random.randn(30, 2) + [2, 2]
        seq3 = np.random.randn(40, 2) - [1, 1]
        
        X = np.vstack([seq1, seq2, seq3])
        lengths = [50, 30, 40]
        
        model = GaussianHMM(n_states=2, random_state=42)
        model.fit(X, lengths=lengths)
        
        assert model.fitted
        predicted = model.predict(X, lengths=lengths)
        assert len(predicted) == sum(lengths)
        
    def test_error_handling(self):
        """Test error handling."""
        model = GaussianHMM(n_states=2)
        
        # Should raise error when predicting before fitting
        with pytest.raises(ValueError):
            model.predict(np.random.randn(10, 2))
            
        # Should raise error when accessing results before fitting
        with pytest.raises(ValueError):
            _ = model.results