"""Base Hidden Markov Model class with common functionality."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
import warnings
from scipy import stats
from hmmlearn import hmm


@dataclass
class HMMResults:
    """Container for HMM results."""
    
    states: np.ndarray
    log_likelihood: float
    transition_matrix: np.ndarray
    initial_probabilities: np.ndarray
    n_iterations: int
    converged: bool
    posterior_probabilities: Optional[np.ndarray] = None
    viterbi_path: Optional[np.ndarray] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    
    def summary(self) -> str:
        """Generate a summary of the HMM results."""
        summary_lines = [
            "Hidden Markov Model Results",
            "===========================",
            f"Number of states: {len(self.initial_probabilities)}",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"Converged: {self.converged}",
            f"Number of iterations: {self.n_iterations}",
        ]
        
        if self.aic is not None:
            summary_lines.append(f"AIC: {self.aic:.4f}")
        if self.bic is not None:
            summary_lines.append(f"BIC: {self.bic:.4f}")
            
        summary_lines.extend([
            "\nInitial state probabilities:",
            str(self.initial_probabilities),
            "\nTransition matrix:",
            str(self.transition_matrix)
        ])
        
        return "\n".join(summary_lines)


class BaseHMM(ABC):
    """Base class for Hidden Markov Models."""
    
    def __init__(
        self,
        n_states: int,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
        init_params: str = "stmc",
        params: str = "stmc",
        covariance_type: str = "diag"
    ):
        """
        Initialize base HMM.
        
        Parameters
        ----------
        n_states : int
            Number of hidden states
        n_iter : int, default=100
            Maximum number of iterations for EM algorithm
        tol : float, default=1e-4
            Convergence tolerance for EM algorithm
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=False
            Whether to print progress during fitting
        init_params : str, default="stmc"
            Controls which parameters are initialized prior to training.
            Can contain any combination of 's' (startprob), 't' (transmat),
            'm' (means), 'c' (covars) for Gaussian models
        params : str, default="stmc"
            Controls which parameters are updated during training.
            Can contain any combination of the same letters as init_params
        covariance_type : str, default="diag"
            Type of covariance parameters to use. Must be one of:
            "spherical", "diag", "full", "tied"
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.init_params = init_params
        self.params = params
        self.covariance_type = covariance_type
        
        # Model will be initialized in subclasses
        self.model = None
        self.fitted = False
        self._results = None
        
    @abstractmethod
    def _create_model(self) -> hmm.BaseHMM:
        """Create the specific HMM model. Must be implemented by subclasses."""
        pass
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None,
        **kwargs
    ) -> 'BaseHMM':
        """
        Fit the HMM model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X. If None, assumes
            X is a single sequence
        **kwargs : additional keyword arguments
            Passed to the underlying hmmlearn fit method
            
        Returns
        -------
        self : BaseHMM
            The fitted model
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Create the model if not already created
        if self.model is None:
            self.model = self._create_model()
            
        # Fit the model
        self.model.fit(X, lengths=lengths)
        self.fitted = True
        
        # Store results
        self._store_results(X, lengths)
        
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Find most likely state sequence using Viterbi algorithm.
        
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
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        states = self.model.predict(X, lengths=lengths)
        # Convert numpy int64 to standard Python int for compatibility
        return states.astype(int)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute posterior probabilities of states.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
            
        Returns
        -------
        posteriors : array, shape (n_samples, n_states)
            State-membership probabilities for each sample
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.model.predict_proba(X, lengths=lengths)
    
    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None
    ) -> float:
        """
        Compute the log-likelihood of X under the model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
            
        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.model.score(X, lengths=lengths)
    
    def sample(
        self,
        n_samples: int = 1,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples from the model.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
        random_state : int, optional
            Random state for reproducibility
            
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix of generated samples
        states : array, shape (n_samples,)
            State sequence of generated samples
        """
        self._check_is_fitted()
        return self.model.sample(n_samples, random_state=random_state)
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution of the Markov chain.
        
        Returns
        -------
        stationary_dist : array, shape (n_states,)
            The stationary distribution
        """
        self._check_is_fitted()
        
        # Get transition matrix
        trans_mat = self.model.transmat_
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)
        
        # Find the eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize to sum to 1
        stationary = stationary / stationary.sum()
        
        return stationary
    
    def decode(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lengths: Optional[List[int]] = None,
        algorithm: str = "viterbi"
    ) -> Tuple[np.ndarray, float]:
        """
        Find the most likely state sequence.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples
        lengths : list of int, optional
            Lengths of the individual sequences in X
        algorithm : {"viterbi", "map"}, default="viterbi"
            Decoder algorithm
            
        Returns
        -------
        states : array, shape (n_samples,)
            Most likely states for each sample
        log_likelihood : float
            Log-likelihood of the produced state sequence
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.model.decode(X, lengths=lengths, algorithm=algorithm)
    
    def _store_results(self, X: np.ndarray, lengths: Optional[List[int]] = None):
        """Store fitting results."""
        # Compute various results
        states = self.predict(X, lengths=lengths)
        log_likelihood = self.score(X, lengths=lengths)
        posterior_probs = self.predict_proba(X, lengths=lengths)
        
        # Calculate information criteria
        n_samples = len(X)
        n_params = self._count_parameters()
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        # Store results
        self._results = HMMResults(
            states=states,
            log_likelihood=log_likelihood,
            transition_matrix=self.model.transmat_,
            initial_probabilities=self.model.startprob_,
            n_iterations=self.model.monitor_.n_iter,
            converged=self.model.monitor_.converged,
            posterior_probabilities=posterior_probs,
            viterbi_path=states,
            aic=aic,
            bic=bic
        )
    
    def _count_parameters(self) -> int:
        """Count the number of free parameters in the model."""
        n_params = 0
        
        # Transition matrix parameters (n_states * (n_states - 1))
        n_params += self.n_states * (self.n_states - 1)
        
        # Initial state probabilities (n_states - 1)
        n_params += self.n_states - 1
        
        # Model-specific parameters (implemented in subclasses)
        n_params += self._count_model_parameters()
        
        return n_params
    
    @abstractmethod
    def _count_model_parameters(self) -> int:
        """Count model-specific parameters. Must be implemented by subclasses."""
        pass
    
    def _check_is_fitted(self):
        """Check if the model has been fitted."""
        if not self.fitted or self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    @property
    def results(self) -> HMMResults:
        """Get fitting results."""
        self._check_is_fitted()
        if self._results is None:
            raise ValueError("No results available. Model may not have been properly fitted.")
        return self._results
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        self._check_is_fitted()
        
        params = {
            'n_states': self.n_states,
            'transition_matrix': self.model.transmat_,
            'initial_probabilities': self.model.startprob_,
            'n_iter': self.model.n_iter_,
            'converged': self.model.monitor_.converged
        }
        
        # Add model-specific parameters
        params.update(self._get_model_params())
        
        return params
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters. Must be implemented by subclasses."""
        pass