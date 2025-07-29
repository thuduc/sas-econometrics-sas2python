"""Tests for Discrete HMM implementation."""

import pytest
import numpy as np
import pandas as pd
from src.models import DiscreteHMM


class TestDiscreteHMM:
    """Test cases for DiscreteHMM."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2-state discrete data."""
        np.random.seed(42)
        
        # State 0: mostly symbols 0, 1
        # State 1: mostly symbols 2, 3
        emission_probs = np.array([
            [0.4, 0.4, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.4]
        ])
        
        n_samples = 500
        states = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        observations = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            observations[i] = np.random.choice(4, p=emission_probs[states[i]])
            
        return observations.reshape(-1, 1), states
    
    def test_initialization(self):
        """Test model initialization."""
        model = DiscreteHMM(n_states=2, n_symbols=4)
        assert model.n_states == 2
        assert model.n_symbols == 4
        assert model.n_features == 1
        assert not model.fitted
        
    def test_fit_predict(self, simple_data):
        """Test fitting and predicting."""
        X, true_states = simple_data
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(X)
        
        assert model.fitted
        assert hasattr(model, 'model')
        assert model.model is not None
        
        # Check emission probabilities shape
        assert model.emissionprob_.shape == (2, 4)
        assert np.allclose(model.emissionprob_.sum(axis=1), 1)
        
        # Predict states
        predicted_states = model.predict(X)
        assert len(predicted_states) == len(X)
        assert set(predicted_states) == {0, 1}
        
    def test_automatic_symbol_detection(self):
        """Test automatic detection of number of symbols."""
        np.random.seed(42)
        X = np.random.randint(0, 6, size=(100, 1))
        
        model = DiscreteHMM(n_states=2, random_state=42)
        model.fit(X)
        
        assert model.n_symbols_ == 6
        assert model.emissionprob_.shape == (2, 6)
        
    def test_symbol_encoding_decoding(self):
        """Test symbol encoding and decoding."""
        model = DiscreteHMM(n_states=2)
        
        # Test encoding
        symbols = ['A', 'B', 'C', 'D']
        X_symbolic = np.array([['A'], ['B'], ['C'], ['A'], ['D']])
        X_encoded = model.decode_symbols(symbols, X_symbolic)
        
        assert X_encoded.shape == X_symbolic.shape
        assert list(X_encoded.flatten()) == [0, 1, 2, 0, 3]
        
        # Test decoding
        X_decoded = model.encode_symbols(symbols, X_encoded)
        assert np.array_equal(X_decoded, X_symbolic)
        
    def test_most_likely_symbols(self, simple_data):
        """Test getting most likely symbols for each state."""
        X, _ = simple_data
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(X)
        
        most_likely = model.most_likely_symbols()
        assert len(most_likely) == 2
        assert all(0 <= s < 4 for s in most_likely)
        
    def test_state_distribution(self, simple_data):
        """Test getting state emission distributions."""
        X, _ = simple_data
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(X)
        
        for state in range(2):
            dist = model.get_state_distribution(state)
            assert 'probabilities' in dist
            assert len(dist['probabilities']) == 4
            assert np.allclose(dist['probabilities'].sum(), 1)
            
    def test_label_switch(self, simple_data):
        """Test label switching functionality."""
        X, _ = simple_data
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(X)
        
        # Get original emission probs
        original_emissions = model.emissionprob_.copy()
        
        # Apply label switching by emission probability of symbol 0
        new_order = model.label_switch(sort_by='emission', symbol=0, ascending=True)
        
        # Check that states are reordered
        assert not np.array_equal(original_emissions, model.emissionprob_)
        
    def test_error_handling(self, simple_data):
        """Test error handling."""
        X, _ = simple_data
        
        # Non-integer data should raise error
        X_float = X.astype(float) + 0.5
        model = DiscreteHMM(n_states=2, n_symbols=4)
        
        with pytest.raises(ValueError):
            model.fit(X_float)
            
        # Out of range symbols
        X_bad = X.copy()
        X_bad[0] = 10  # Symbol out of range
        
        with pytest.raises(ValueError):
            model.fit(X_bad)
            
    def test_with_pandas(self, simple_data):
        """Test with pandas DataFrame input."""
        X, _ = simple_data
        df = pd.DataFrame(X, columns=['symbol'])
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(df)
        
        predicted = model.predict(df)
        assert len(predicted) == len(df)
        
    def test_multiple_sequences(self):
        """Test with multiple sequences."""
        np.random.seed(42)
        
        # Generate 3 sequences
        seq1 = np.random.randint(0, 3, size=(50, 1))
        seq2 = np.random.randint(1, 4, size=(30, 1))
        seq3 = np.random.randint(0, 4, size=(40, 1))
        
        X = np.vstack([seq1, seq2, seq3])
        lengths = [50, 30, 40]
        
        model = DiscreteHMM(n_states=2, n_symbols=4, random_state=42)
        model.fit(X, lengths=lengths)
        
        assert model.fitted
        predicted = model.predict(X, lengths=lengths)
        assert len(predicted) == sum(lengths)