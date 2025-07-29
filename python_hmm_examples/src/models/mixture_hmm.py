"""Gaussian Mixture Hidden Markov Model implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List, Tuple
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import warnings

from .base_hmm import BaseHMM


class GaussianMixtureHMM(BaseHMM):
    """
    Hidden Markov Model with Gaussian Mixture emissions.
    
    This model assumes that the observations in each state are generated
    from a mixture of Gaussian distributions.
    
    Parameters
    ----------
    n_states : int
        Number of hidden states
    n_components : int, default=1
        Number of mixture components per state
    n_features : int, optional
        Number of features in the observations. If not specified, will be
        inferred from the data during fitting
    covariance_type : str, default="diag"
        Type of covariance parameters to use. Must be one of:
        - "spherical": each component has a single variance
        - "diag": each component has a diagonal covariance matrix
        - "full": each component has a full covariance matrix
        - "tied": all components share the same covariance matrix
    min_covar : float, default=1e-3
        Floor on the diagonal of the covariance matrix to prevent
        overfitting
    weights_prior : float, default=1.0
        Prior for the mixture component weights
    means_prior : array-like, optional
        Prior means for the Gaussian mixture components
    means_weight : float, default=0
        Weight of the means prior
    covars_prior : array-like, optional
        Prior covariance for the Gaussian mixture components
    covars_weight : float, default=1
        Weight of the covariance prior
    **kwargs : additional keyword arguments
        Passed to the BaseHMM constructor
        
    Attributes
    ----------
    weights_ : array, shape (n_states, n_components)
        Mixture weights for each state
    means_ : array, shape (n_states, n_components, n_features)
        Mean parameters for each mixture component in each state
    covars_ : array
        Covariance parameters for each mixture component. Shape depends
        on covariance_type
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import GaussianMixtureHMM
    >>> 
    >>> # Generate synthetic data with mixture components
    >>> np.random.seed(42)
    >>> n_samples = 1000
    >>> 
    >>> # State 1: mixture of two Gaussians
    >>> state1_samples = 400
    >>> comp1 = np.random.randn(state1_samples // 2, 2) + [0, 0]
    >>> comp2 = np.random.randn(state1_samples // 2, 2) + [3, 3]
    >>> state1_data = np.vstack([comp1, comp2])
    >>> 
    >>> # State 2: mixture of two Gaussians
    >>> state2_samples = 600
    >>> comp3 = np.random.randn(state2_samples // 2, 2) + [-3, -3]
    >>> comp4 = np.random.randn(state2_samples // 2, 2) + [0, -3]
    >>> state2_data = np.vstack([comp3, comp4])
    >>> 
    >>> X = np.vstack([state1_data, state2_data])
    >>> 
    >>> # Fit model
    >>> model = GaussianMixtureHMM(n_states=2, n_components=2)
    >>> model.fit(X)
    >>> 
    >>> # Predict states
    >>> predicted_states = model.predict(X)
    """
    
    def __init__(
        self,
        n_states: int,
        n_components: int = 1,
        n_features: Optional[int] = None,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        weights_prior: float = 1.0,
        means_prior: Optional[np.ndarray] = None,
        means_weight: float = 0,
        covars_prior: Optional[Union[float, np.ndarray]] = None,
        covars_weight: float = 1,
        **kwargs
    ):
        """Initialize Gaussian Mixture HMM."""
        super().__init__(n_states=n_states, covariance_type=covariance_type, **kwargs)
        
        self.n_components = n_components
        self.n_features = n_features
        self.min_covar = min_covar
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        
    def _create_model(self) -> hmm.GMMHMM:
        """Create the Gaussian Mixture HMM model."""
        model = hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_components,
            covariance_type=self.covariance_type,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            init_params=self.init_params,
            params=self.params
        )
        
        return model
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None,
        init_weights: Optional[np.ndarray] = None,
        init_means: Optional[np.ndarray] = None,
        init_covars: Optional[np.ndarray] = None,
        init_startprob: Optional[np.ndarray] = None,
        init_transmat: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'GaussianMixtureHMM':
        """
        Fit the Gaussian Mixture HMM model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
        init_weights : array-like, shape (n_states, n_components), optional
            Initial mixture weights for each state
        init_means : array-like, shape (n_states, n_components, n_features), optional
            Initial means for the mixture components
        init_covars : array-like, optional
            Initial covariances for the mixture components
        init_startprob : array-like, shape (n_states,), optional
            Initial starting probabilities
        init_transmat : array-like, shape (n_states, n_states), optional
            Initial transition matrix
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : GaussianMixtureHMM
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
        if init_weights is not None:
            self.model.weights_ = init_weights
            self.model.init_params = self.model.init_params.replace('w', '')
            
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
        """Count Gaussian Mixture-specific parameters."""
        if self.n_features is None:
            warnings.warn("n_features not set, cannot count parameters accurately")
            return 0
            
        n_params = 0
        
        # Mixture weights (n_states * (n_components - 1))
        n_params += self.n_states * (self.n_components - 1)
        
        # Mean parameters (n_states * n_components * n_features)
        n_params += self.n_states * self.n_components * self.n_features
        
        # Covariance parameters
        if self.covariance_type == "spherical":
            # One variance per component per state
            n_params += self.n_states * self.n_components
        elif self.covariance_type == "diag":
            # Diagonal covariance per component per state
            n_params += self.n_states * self.n_components * self.n_features
        elif self.covariance_type == "full":
            # Full covariance matrix per component per state
            n_params += self.n_states * self.n_components * self.n_features * (self.n_features + 1) // 2
        elif self.covariance_type == "tied":
            # One shared covariance matrix per state
            n_params += self.n_states * self.n_features * (self.n_features + 1) // 2
            
        return n_params
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get Gaussian Mixture-specific parameters."""
        params = {
            'weights': self.model.weights_,
            'means': self.model.means_,
            'covariances': self.model.covars_,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type
        }
        return params
    
    @property
    def weights_(self) -> np.ndarray:
        """Get the mixture weight parameters."""
        self._check_is_fitted()
        return self.model.weights_
    
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
    
    def get_state_distribution(self, state: int) -> Dict[str, Any]:
        """
        Get the Gaussian mixture distribution parameters for a specific state.
        
        Parameters
        ----------
        state : int
            The state index
            
        Returns
        -------
        dist_params : dict
            Dictionary containing 'weights', 'means', and 'covariances' 
            for the mixture components in the state
        """
        self._check_is_fitted()
        
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State must be between 0 and {self.n_states-1}")
            
        weights = self.weights_[state]
        means = self.means_[state]
        
        # Extract covariances based on type
        if self.covariance_type == "spherical":
            covs = []
            for comp in range(self.n_components):
                cov = np.eye(self.n_features) * self.covars_[state, comp]
                covs.append(cov)
        elif self.covariance_type == "diag":
            covs = []
            for comp in range(self.n_components):
                cov = np.diag(self.covars_[state, comp])
                covs.append(cov)
        elif self.covariance_type == "full":
            covs = self.covars_[state]
        elif self.covariance_type == "tied":
            covs = [self.covars_[state] for _ in range(self.n_components)]
            
        return {
            'weights': weights,
            'means': means,
            'covariances': np.array(covs)
        }
    
    def predict_component(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both states and mixture components for each observation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
            
        Returns
        -------
        states : array, shape (n_samples,)
            Most likely states for each sample
        components : array, shape (n_samples,)
            Most likely mixture component within each state
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Get state predictions
        states = self.predict(X, lengths=lengths)
        
        # Get component predictions within each state
        components = np.zeros(len(X), dtype=int)
        
        for state in range(self.n_states):
            mask = states == state
            if np.any(mask):
                X_state = X[mask]
                
                # Create a Gaussian mixture for this state
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    random_state=self.random_state
                )
                
                # Set parameters from the fitted model
                gmm.weights_ = self.weights_[state]
                gmm.means_ = self.means_[state]
                gmm.covariances_ = self._get_state_covariances(state)
                gmm.precisions_cholesky_ = gmm._compute_precision_cholesky(
                    gmm.covariances_, self.covariance_type
                )
                
                # Predict components
                components[mask] = gmm.predict(X_state)
                
        return states, components
    
    def _get_state_covariances(self, state: int) -> np.ndarray:
        """Extract covariances for a specific state."""
        if self.covariance_type == "spherical":
            return self.covars_[state]
        elif self.covariance_type == "diag":
            return self.covars_[state]
        elif self.covariance_type == "full":
            return self.covars_[state]
        elif self.covariance_type == "tied":
            return self.covars_[state]
    
    def label_switch(
        self,
        sort_by: str = 'means',
        ascending: bool = True,
        component_sort: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reorder states and components based on a criterion to handle label switching.
        
        Parameters
        ----------
        sort_by : str, default='means'
            Criterion to sort states by. Options:
            - 'means': Sort by the average of component means
            - 'weights': Sort by the weight of the first component
        ascending : bool, default=True
            Whether to sort in ascending order
        component_sort : bool, default=True
            Whether to also sort components within each state
            
        Returns
        -------
        state_order : array, shape (n_states,)
            The new ordering of states
        component_orders : array, shape (n_states, n_components)
            The new ordering of components within each state
        """
        self._check_is_fitted()
        
        # Sort states
        if sort_by == 'means':
            # Sort by average of all component means in first dimension
            values = np.mean(self.means_[:, :, 0], axis=1)
        elif sort_by == 'weights':
            # Sort by weight of first component
            values = self.weights_[:, 0]
        else:
            raise ValueError(f"Unknown sort_by criterion: {sort_by}")
            
        if ascending:
            state_order = np.argsort(values)
        else:
            state_order = np.argsort(-values)
            
        # Sort components within each state if requested
        component_orders = np.zeros((self.n_states, self.n_components), dtype=int)
        
        if component_sort:
            for s in range(self.n_states):
                # Sort components by their means in first dimension
                comp_values = self.means_[s, :, 0]
                if ascending:
                    component_orders[s] = np.argsort(comp_values)
                else:
                    component_orders[s] = np.argsort(-comp_values)
        else:
            for s in range(self.n_states):
                component_orders[s] = np.arange(self.n_components)
                
        # Reorder model parameters
        # First reorder components within states
        new_weights = np.zeros_like(self.model.weights_)
        new_means = np.zeros_like(self.model.means_)
        
        for s in range(self.n_states):
            comp_order = component_orders[s]
            new_weights[s] = self.model.weights_[s][comp_order]
            new_means[s] = self.model.means_[s][comp_order]
            
        # Then reorder states
        self.model.weights_ = new_weights[state_order]
        self.model.means_ = new_means[state_order]
        
        # Reorder covariances
        if self.covariance_type in ["spherical", "diag"]:
            new_covars = np.zeros_like(self.model.covars_)
            for s in range(self.n_states):
                comp_order = component_orders[s]
                new_covars[s] = self.model.covars_[s][comp_order]
            self.model.covars_ = new_covars[state_order]
        elif self.covariance_type == "full":
            new_covars = np.zeros_like(self.model.covars_)
            for s in range(self.n_states):
                comp_order = component_orders[s]
                new_covars[s] = self.model.covars_[s][comp_order]
            self.model.covars_ = new_covars[state_order]
        elif self.covariance_type == "tied":
            self.model.covars_ = self.model.covars_[state_order]
            
        # Reorder transition matrix
        self.model.transmat_ = self.model.transmat_[state_order][:, state_order]
        
        # Reorder initial probabilities
        self.model.startprob_ = self.model.startprob_[state_order]
        
        return state_order, component_orders