"""Tests for business cycle analysis example."""

import pytest
import numpy as np
import pandas as pd
from src.examples.business_cycle import (
    create_gnp_data,
    create_nber_recession_data,
    fit_business_cycle_hmm
)


class TestBusinessCycle:
    """Test business cycle analysis functionality."""
    
    def test_create_gnp_data(self):
        """Test GNP data creation."""
        df = create_gnp_data()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert 'date' in df.columns
        assert 'gnp' in df.columns
        assert 'dgnp' in df.columns
        
        # Check data dimensions (135 quarters minus 1 for growth rate calculation)
        assert len(df) == 135
        
        # Check date range
        assert df['date'].min() == pd.Timestamp('1951-04-01')
        assert df['date'].max() == pd.Timestamp('1984-10-01')
        
        # Check GNP values
        assert df['gnp'].min() > 0
        assert df['gnp'].max() > df['gnp'].min()
        
        # Check growth rate calculation
        assert not df['dgnp'].isna().any()
        
    def test_create_nber_recession_data(self):
        """Test NBER recession data creation."""
        df = create_nber_recession_data()
        
        # Check dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert 'date' in df.columns
        assert 'recessionNBER' in df.columns
        
        # Check data dimensions
        assert len(df) == 135  # Matches GNP data after removing first row
        
        # Check recession indicator
        assert df['recessionNBER'].isin([0, 1]).all()
        
        # Check that we have some recessions
        assert df['recessionNBER'].sum() > 0
        assert df['recessionNBER'].sum() < len(df)  # Not all periods are recessions
        
    def test_business_cycle_hmm_basic(self):
        """Test basic HMM fitting for business cycle."""
        # Create data
        df = create_gnp_data()
        
        # Fit model without initial parameters
        model, filtered_probs, smoothed_probs, states = fit_business_cycle_hmm(
            df, use_initial_params=False
        )
        
        # Check model outputs
        assert model is not None
        assert hasattr(model, 'n_states')
        assert model.n_states == 2
        
        # Check probability matrices
        n_obs = len(df) - 4  # Adjusted for AR(4) lags
        assert filtered_probs.shape == (n_obs, 2)
        assert smoothed_probs.shape == (n_obs, 2)
        
        # Check probabilities sum to 1
        np.testing.assert_allclose(filtered_probs.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(smoothed_probs.sum(axis=1), 1.0, rtol=1e-5)
        
        # Check states
        assert states.shape == (n_obs,)
        assert np.all(np.isin(states, [0, 1]))
        
    def test_business_cycle_hmm_with_initial_params(self):
        """Test HMM fitting with initial parameters."""
        # Create data
        df = create_gnp_data()
        
        # Fit model with initial parameters
        model, filtered_probs, smoothed_probs, states = fit_business_cycle_hmm(
            df, use_initial_params=True
        )
        
        # Check that initial parameters were used
        assert model.transition_matrix is not None
        assert model.transition_matrix.shape == (2, 2)
        
        # Check transition matrix properties
        np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all(model.transition_matrix >= 0)
        assert np.all(model.transition_matrix <= 1)
        
        # Check AR coefficients
        assert model.ar_coefficients is not None
        assert model.ar_coefficients.shape == (2, 4)  # 2 states, AR(4)
        
        # Check intercepts and variances
        assert len(model.intercepts) == 2
        assert len(model.variances) == 2
        assert np.all(model.variances > 0)
        
    def test_state_persistence(self):
        """Test state persistence calculation."""
        # Create and fit model
        df = create_gnp_data()
        model, _, _, _ = fit_business_cycle_hmm(df, use_initial_params=True)
        
        # Calculate expected durations
        for i in range(2):
            persistence = 1 / (1 - model.transition_matrix[i, i])
            assert persistence > 0
            assert persistence < 100  # Reasonable bound
            
    def test_data_alignment(self):
        """Test that GNP and NBER data align properly."""
        gnp_df = create_gnp_data()
        nber_df = create_nber_recession_data()
        
        # Check that dates align
        assert len(gnp_df) == len(nber_df)
        assert (gnp_df['date'].values == nber_df['date'].values).all()