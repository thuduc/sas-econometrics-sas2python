"""Poisson Hidden Markov Model implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from scipy import stats
from hmmlearn.base import BaseHMM as HMMBase
from sklearn.utils import check_random_state
import warnings

from .base_hmm import BaseHMM, HMMResults


class PoissonHMM(BaseHMM):
    """
    Hidden Markov Model with Poisson emissions.
    
    This model assumes that the observations are count data generated from
    a Poisson distribution whose rate parameter depends on the hidden state.
    
    Parameters
    ----------
    n_states : int
        Number of hidden states
    n_features : int, default=1
        Number of features (dimensions) in the observations
    min_lambda : float, default=1e-3
        Minimum value for lambda (rate) parameters to ensure numerical stability
    **kwargs : additional keyword arguments
        Passed to the BaseHMM constructor
        
    Attributes
    ----------
    lambdas_ : array, shape (n_states, n_features)
        Rate parameters for each state and feature
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import PoissonHMM
    >>> 
    >>> # Generate synthetic count data
    >>> np.random.seed(42)
    >>> 
    >>> # Define rate parameters for each state
    >>> lambdas = np.array([[2.0], [10.0]])  # Low and high count states
    >>> 
    >>> # Generate sequence
    >>> states = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    >>> X = []
    >>> for state in states:
    ...     count = np.random.poisson(lambdas[state])
    ...     X.append(count)
    >>> X = np.array(X).reshape(-1, 1)
    >>> 
    >>> # Fit model
    >>> model = PoissonHMM(n_states=2)
    >>> model.fit(X)
    >>> 
    >>> # Predict states
    >>> predicted_states = model.predict(X)
    """
    
    def __init__(
        self,
        n_states: int,
        n_features: int = 1,
        min_lambda: float = 1e-3,
        **kwargs
    ):
        """Initialize Poisson HMM."""
        # Set Poisson-specific default params
        if 'init_params' not in kwargs:
            kwargs['init_params'] = 'stl'
        if 'params' not in kwargs:
            kwargs['params'] = 'stl'
            
        super().__init__(n_states=n_states, **kwargs)
        
        self.n_features = n_features
        self.min_lambda = min_lambda
        
    def _create_model(self) -> '_PoissonHMM':
        """Create the Poisson HMM model."""
        model = _PoissonHMM(
            n_components=self.n_states,
            n_features=self.n_features,
            min_lambda=self.min_lambda,
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
        init_lambdas: Optional[np.ndarray] = None,
        init_startprob: Optional[np.ndarray] = None,
        init_transmat: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'PoissonHMM':
        """
        Fit the Poisson HMM model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples. Values should be non-negative
            integers representing counts
        lengths : list of int, optional
            Lengths of the individual sequences in X
        init_lambdas : array-like, shape (n_states, n_features), optional
            Initial rate parameters
        init_startprob : array-like, shape (n_states,), optional
            Initial starting probabilities
        init_transmat : array-like, shape (n_states, n_states), optional
            Initial transition matrix
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : PoissonHMM
            The fitted model
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Check that X contains non-negative integers
        if np.any(X < 0):
            raise ValueError("Poisson HMM requires non-negative count data")
            
        if not np.allclose(X, X.astype(int)):
            warnings.warn("Poisson HMM expects integer count data. "
                        "Non-integer values will be rounded.")
            X = np.round(X).astype(int)
            
        # Check n_features
        if X.shape[1] != self.n_features:
            warnings.warn(f"n_features was {self.n_features} but data has {X.shape[1]} features. "
                        f"Updating n_features to {X.shape[1]}")
            self.n_features = X.shape[1]
            
        # Create the model if not already created
        if self.model is None:
            self.model = self._create_model()
            
        # Set initial parameters if provided
        if init_lambdas is not None:
            self.model.lambdas_ = init_lambdas
            self.model.init_params = self.model.init_params.replace('l', '')
            
        if init_startprob is not None:
            self.model.startprob_ = init_startprob
            self.model.init_params = self.model.init_params.replace('s', '')
            
        if init_transmat is not None:
            self.model.transmat_ = init_transmat
            self.model.init_params = self.model.init_params.replace('t', '')
            
        # Call parent fit method
        return super().fit(X, lengths=lengths, **kwargs)
    
    def _count_model_parameters(self) -> int:
        """Count Poisson-specific parameters."""
        # Rate parameters (n_states * n_features)
        return self.n_states * self.n_features
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get Poisson-specific parameters."""
        params = {
            'lambdas': self.model.lambdas_,
            'n_features': self.n_features
        }
        return params
    
    @property
    def lambdas_(self) -> np.ndarray:
        """Get the rate parameters."""
        self._check_is_fitted()
        return self.model.lambdas_
    
    def get_state_distribution(self, state: int) -> Dict[str, np.ndarray]:
        """
        Get the Poisson distribution parameters for a specific state.
        
        Parameters
        ----------
        state : int
            The state index
            
        Returns
        -------
        dist_params : dict
            Dictionary containing 'lambda' (rate parameters) for the state
        """
        self._check_is_fitted()
        
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State must be between 0 and {self.n_states-1}")
            
        return {'lambda': self.lambdas_[state]}
    
    def label_switch(self, sort_by: str = 'lambda', feature: int = 0, ascending: bool = True) -> np.ndarray:
        """
        Reorder states based on a criterion to handle label switching.
        
        Parameters
        ----------
        sort_by : str, default='lambda'
            Criterion to sort states by. Options:
            - 'lambda': Sort by rate parameter
            - 'mean': Sort by mean (same as lambda for Poisson)
        feature : int, default=0
            Feature index to use for sorting when n_features > 1
        ascending : bool, default=True
            Whether to sort in ascending order
            
        Returns
        -------
        new_order : array, shape (n_states,)
            The new ordering of states
        """
        self._check_is_fitted()
        
        if sort_by in ['lambda', 'mean']:
            # Sort by lambda parameter for specified feature
            values = self.lambdas_[:, feature]
        else:
            raise ValueError(f"Unknown sort_by criterion: {sort_by}")
            
        if ascending:
            new_order = np.argsort(values)
        else:
            new_order = np.argsort(-values)
            
        # Reorder model parameters
        self.model.lambdas_ = self.model.lambdas_[new_order]
        
        # Reorder transition matrix
        self.model.transmat_ = self.model.transmat_[new_order][:, new_order]
        
        # Reorder initial probabilities
        self.model.startprob_ = self.model.startprob_[new_order]
        
        return new_order


class _PoissonHMM(HMMBase):
    """
    Internal Poisson HMM implementation compatible with hmmlearn.
    
    This class implements the core Poisson HMM functionality following
    the hmmlearn API conventions.
    """
    
    def __init__(
        self,
        n_components: int = 1,
        n_features: int = 1,
        min_lambda: float = 1e-3,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        algorithm: str = "viterbi",
        random_state: Optional[int] = None,
        n_iter: int = 10,
        tol: float = 1e-2,
        verbose: bool = False,
        params: str = "stl",
        init_params: str = "stl",
        implementation: str = "log"
    ):
        """Initialize internal Poisson HMM."""
        super().__init__(
            n_components=n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation
        )
        
        self.n_features = n_features
        self.min_lambda = min_lambda
        
    def _init(self, X, lengths=None):
        """Initialize model parameters."""
        super()._init(X, lengths=lengths)
        
        n_samples, n_features = X.shape
        
        if 'l' in self.init_params:
            # Initialize lambdas using k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_components, 
                          random_state=self.random_state)
            kmeans.fit(X)
            
            # Use cluster means as initial lambdas
            self.lambdas_ = np.maximum(kmeans.cluster_centers_, self.min_lambda)
            
    def _check(self):
        """Validate model parameters."""
        super()._check()
        
        # Check lambdas
        self.lambdas_ = np.asarray(self.lambdas_)
        if self.lambdas_.shape != (self.n_components, self.n_features):
            raise ValueError(
                f"lambdas_ must have shape (n_components, n_features), "
                f"got {self.lambdas_.shape}"
            )
            
        # Ensure lambdas are positive
        self.lambdas_ = np.maximum(self.lambdas_, self.min_lambda)
        
    def _compute_log_likelihood(self, X):
        """Compute the log likelihood under the model."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            for j in range(n_features):
                log_prob[:, i] += stats.poisson.logpmf(X[:, j], self.lambdas_[i, j])
                
        return log_prob
    
    def _generate_sample_from_state(self, state, random_state=None):
        """Generate a random sample from a given state."""
        rng = check_random_state(random_state)
        return rng.poisson(self.lambdas_[state])
    
    def _do_mstep(self, stats):
        """M-step of the EM algorithm."""
        super()._do_mstep(stats)
        
        if 'l' in self.params:
            # Update lambdas
            lambdas_num = stats['obs_for_lambdas']
            lambdas_den = stats['post_sum']
            
            # Avoid division by zero
            lambdas_den = np.maximum(lambdas_den, 1e-10)
            
            self.lambdas_ = (lambdas_num / lambdas_den[:, np.newaxis])
            self.lambdas_ = np.maximum(self.lambdas_, self.min_lambda)
            
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                        posteriors, fwdlattice, bwdlattice):
        """Update sufficient statistics from a pass through the data."""
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice
        )
        
        if 'l' in self.params:
            # Accumulate statistics for lambda updates
            if 'obs_for_lambdas' not in stats:
                stats['obs_for_lambdas'] = np.zeros((self.n_components, self.n_features))
                stats['post_sum'] = np.zeros(self.n_components)
                
            for t in range(X.shape[0]):
                for i in range(self.n_components):
                    stats['obs_for_lambdas'][i] += posteriors[t, i] * X[t]
                    
            stats['post_sum'] += posteriors.sum(axis=0)
                    
    def _get_n_fit_scalars_per_param(self):
        """Get the number of scalars for each parameter type."""
        return {
            's': self.n_components - 1,  # startprob
            't': self.n_components * (self.n_components - 1),  # transmat  
            'l': self.n_components * self.n_features  # lambdas
        }