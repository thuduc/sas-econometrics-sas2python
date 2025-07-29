"""Gaussian Hidden Markov Model implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from hmmlearn import hmm
import warnings

from .base_hmm import BaseHMM


class GaussianHMM(BaseHMM):
    """
    Hidden Markov Model with Gaussian emissions.
    
    This model assumes that the observations are generated from a Gaussian
    distribution whose parameters depend on the hidden state.
    
    Parameters
    ----------
    n_states : int
        Number of hidden states
    n_features : int, optional
        Number of features in the observations. If not specified, will be
        inferred from the data during fitting
    covariance_type : str, default="diag"
        Type of covariance parameters to use. Must be one of:
        - "spherical": each state uses a single variance value
        - "diag": each state uses a diagonal covariance matrix
        - "full": each state uses a full covariance matrix
        - "tied": all states share the same full covariance matrix
    min_covar : float, default=1e-3
        Floor on the diagonal of the covariance matrix to prevent
        overfitting
    means_prior : array-like, optional
        Prior means for the Gaussian states
    means_weight : float, default=0
        Weight of the means prior
    covars_prior : array-like, optional
        Prior covariance for the Gaussian states
    covars_weight : float, default=1
        Weight of the covariance prior
    **kwargs : additional keyword arguments
        Passed to the BaseHMM constructor
        
    Attributes
    ----------
    means_ : array, shape (n_states, n_features)
        Mean parameters for each state
    covars_ : array
        Covariance parameters for each state. The shape depends on
        covariance_type
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import GaussianHMM
    >>> 
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> states = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    >>> means = np.array([[0, 0], [5, 5]])
    >>> X = np.random.randn(100, 2) + means[states]
    >>> 
    >>> # Fit model
    >>> model = GaussianHMM(n_states=2)
    >>> model.fit(X)
    >>> 
    >>> # Predict states
    >>> predicted_states = model.predict(X)
    """
    
    def __init__(
        self,
        n_states: int,
        n_features: Optional[int] = None,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        means_prior: Optional[np.ndarray] = None,
        means_weight: float = 0,
        covars_prior: Optional[Union[float, np.ndarray]] = None,
        covars_weight: float = 1,
        **kwargs
    ):
        """Initialize Gaussian HMM."""
        super().__init__(n_states=n_states, covariance_type=covariance_type, **kwargs)
        
        self.n_features = n_features
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        
    def _create_model(self) -> hmm.GaussianHMM:
        """Create the Gaussian HMM model."""
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            init_params=self.init_params,
            params=self.params
        )
        
        # Set priors if provided
        if self.means_prior is not None:
            model.means_prior = self.means_prior
            model.means_weight = self.means_weight
            
        if self.covars_prior is not None:
            model.covars_prior = self.covars_prior
            model.covars_weight = self.covars_weight
            
        return model
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None,
        init_means: Optional[np.ndarray] = None,
        init_covars: Optional[np.ndarray] = None,
        init_startprob: Optional[np.ndarray] = None,
        init_transmat: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'GaussianHMM':
        """
        Fit the Gaussian HMM model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
        init_means : array-like, shape (n_states, n_features), optional
            Initial means for the states
        init_covars : array-like, optional
            Initial covariances for the states
        init_startprob : array-like, shape (n_states,), optional
            Initial starting probabilities
        init_transmat : array-like, shape (n_states, n_states), optional
            Initial transition matrix
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : GaussianHMM
            The fitted model
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Infer n_features if not set
        if self.n_features is None:
            self.n_features = X.shape[1]
        elif X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            
        # Create the model if not already created
        if self.model is None:
            self.model = self._create_model()
            
        # Set initial parameters if provided
        if init_means is not None:
            self.model.means_ = init_means
            self.model.init_params = self.model.init_params.replace('m', '')
            
        if init_covars is not None:
            self.model.covars_ = init_covars
            self.model.init_params = self.model.init_params.replace('c', '')
            
        if init_startprob is not None:
            self.model.startprob_ = init_startprob
            self.model.init_params = self.model.init_params.replace('s', '')
            
        if init_transmat is not None:
            self.model.transmat_ = init_transmat
            self.model.init_params = self.model.init_params.replace('t', '')
            
        # Call parent fit method
        return super().fit(X, lengths=lengths, **kwargs)
    
    def _count_model_parameters(self) -> int:
        """Count Gaussian-specific parameters."""
        if self.n_features is None:
            warnings.warn("n_features not set, cannot count parameters accurately")
            return 0
            
        n_params = 0
        
        # Mean parameters (n_states * n_features)
        n_params += self.n_states * self.n_features
        
        # Covariance parameters
        if self.covariance_type == "spherical":
            # One variance per state
            n_params += self.n_states
        elif self.covariance_type == "diag":
            # Diagonal covariance per state
            n_params += self.n_states * self.n_features
        elif self.covariance_type == "full":
            # Full covariance matrix per state
            n_params += self.n_states * self.n_features * (self.n_features + 1) // 2
        elif self.covariance_type == "tied":
            # One shared covariance matrix
            n_params += self.n_features * (self.n_features + 1) // 2
            
        return n_params
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get Gaussian-specific parameters."""
        params = {
            'means': self.model.means_,
            'covariances': self.model.covars_,
            'covariance_type': self.covariance_type
        }
        return params
    
    @property
    def means_(self) -> np.ndarray:
        """Get the mean parameters."""
        self._check_is_fitted()
        return self.model.means_
    
    @property
    def covars_(self) -> np.ndarray:
        """Get the covariance parameters."""
        self._check_is_fitted()
        return self.model.covars_
    
    def get_state_distribution(self, state: int) -> Dict[str, np.ndarray]:
        """
        Get the Gaussian distribution parameters for a specific state.
        
        Parameters
        ----------
        state : int
            The state index
            
        Returns
        -------
        dist_params : dict
            Dictionary containing 'mean' and 'covariance' for the state
        """
        self._check_is_fitted()
        
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State must be between 0 and {self.n_states-1}")
            
        mean = self.means_[state]
        
        if self.covariance_type == "spherical":
            cov = np.eye(self.n_features) * self.covars_[state]
        elif self.covariance_type == "diag":
            cov = np.diag(self.covars_[state])
        elif self.covariance_type == "full":
            cov = self.covars_[state]
        elif self.covariance_type == "tied":
            cov = self.covars_
            
        return {'mean': mean, 'covariance': cov}
    
    def label_switch(self, sort_by: str = 'means', ascending: bool = True) -> np.ndarray:
        """
        Reorder states based on a criterion to handle label switching.
        
        Parameters
        ----------
        sort_by : str, default='means'
            Criterion to sort states by. Options:
            - 'means': Sort by the first dimension of means
            - 'variance': Sort by average variance
        ascending : bool, default=True
            Whether to sort in ascending order
            
        Returns
        -------
        new_order : array, shape (n_states,)
            The new ordering of states
        """
        self._check_is_fitted()
        
        if sort_by == 'means':
            # Sort by first dimension of means
            values = self.means_[:, 0]
        elif sort_by == 'variance':
            # Sort by average variance
            if self.covariance_type == "spherical":
                values = self.covars_
            elif self.covariance_type == "diag":
                values = np.mean(self.covars_, axis=1)
            elif self.covariance_type == "full":
                values = np.array([np.trace(cov) for cov in self.covars_])
            elif self.covariance_type == "tied":
                # All states have same covariance, return original order
                return np.arange(self.n_states)
        else:
            raise ValueError(f"Unknown sort_by criterion: {sort_by}")
            
        if ascending:
            new_order = np.argsort(values)
        else:
            new_order = np.argsort(-values)
            
        # Reorder model parameters
        self.model.means_ = self.model.means_[new_order]
        
        # Work with internal covariance representation
        if self.covariance_type == "spherical":
            self.model._covars_ = self.model._covars_[new_order]
        elif self.covariance_type == "diag":
            self.model._covars_ = self.model._covars_[new_order]
        elif self.covariance_type == "full":
            self.model._covars_ = self.model._covars_[new_order]
            
        # Reorder transition matrix
        self.model.transmat_ = self.model.transmat_[new_order][:, new_order]
        
        # Reorder initial probabilities
        self.model.startprob_ = self.model.startprob_[new_order]
        
        return new_order