"""Discrete/Multinomial Hidden Markov Model implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
from hmmlearn import hmm
import warnings

from .base_hmm import BaseHMM


class DiscreteHMM(BaseHMM):
    """
    Hidden Markov Model with discrete/multinomial emissions.
    
    This model assumes that the observations are discrete symbols from a
    finite alphabet, with emission probabilities depending on the hidden state.
    
    Parameters
    ----------
    n_states : int
        Number of hidden states
    n_features : int, default=1
        Number of features (dimensions) in the observations
    n_symbols : int or list of int, optional
        Number of possible symbols for each feature. If an int, the same
        number of symbols is assumed for all features. If a list, must
        have length n_features. If not specified, will be inferred from
        the data during fitting
    **kwargs : additional keyword arguments
        Passed to the BaseHMM constructor
        
    Attributes
    ----------
    emissionprob_ : array, shape (n_states, n_symbols) or list of arrays
        Emission probability distributions for each state. If n_features > 1,
        this is a list of arrays, one for each feature
    n_symbols_ : int or list of int
        Number of symbols for each feature
    
    Examples
    --------
    >>> import numpy as np
    >>> from src.models import DiscreteHMM
    >>> 
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> 
    >>> # Define emission probabilities for each state
    >>> emission_probs = np.array([
    ...     [0.7, 0.2, 0.1],  # State 0: mostly symbol 0
    ...     [0.1, 0.2, 0.7]   # State 1: mostly symbol 2
    ... ])
    >>> 
    >>> # Generate sequence
    >>> states = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
    >>> X = []
    >>> for state in states:
    ...     symbol = np.random.choice(3, p=emission_probs[state])
    ...     X.append(symbol)
    >>> X = np.array(X).reshape(-1, 1)
    >>> 
    >>> # Fit model
    >>> model = DiscreteHMM(n_states=2, n_symbols=3)
    >>> model.fit(X)
    >>> 
    >>> # Predict states
    >>> predicted_states = model.predict(X)
    """
    
    def __init__(
        self,
        n_states: int,
        n_features: int = 1,
        n_symbols: Optional[Union[int, List[int]]] = None,
        **kwargs
    ):
        """Initialize Discrete HMM."""
        # Remove init_params and params from kwargs if they exist
        # We'll set them specifically for discrete models
        init_params = kwargs.pop('init_params', 'ste')
        params = kwargs.pop('params', 'ste')
        
        super().__init__(
            n_states=n_states,
            init_params=init_params,
            params=params,
            **kwargs
        )
        
        self.n_features = n_features
        self.n_symbols = n_symbols
        self.n_symbols_ = None
        
    def _create_model(self) -> hmm.CategoricalHMM:
        """Create the Discrete/Categorical HMM model."""
        if self.n_symbols_ is None:
            raise ValueError("n_symbols must be set before creating model. Call fit() first.")
            
        model = hmm.CategoricalHMM(
            n_components=self.n_states,
            n_features=self.n_symbols_,
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
        init_emissionprob: Optional[np.ndarray] = None,
        init_startprob: Optional[np.ndarray] = None,
        init_transmat: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'DiscreteHMM':
        """
        Fit the Discrete HMM model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples. Values should be integers
            representing discrete symbols
        lengths : list of int, optional
            Lengths of the individual sequences in X
        init_emissionprob : array-like, shape (n_states, n_symbols), optional
            Initial emission probabilities
        init_startprob : array-like, shape (n_states,), optional
            Initial starting probabilities
        init_transmat : array-like, shape (n_states, n_states), optional
            Initial transition matrix
        **kwargs : additional keyword arguments
            Passed to the parent fit method
            
        Returns
        -------
        self : DiscreteHMM
            The fitted model
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Check that X contains integers
        if not np.issubdtype(X.dtype, np.integer):
            # Check if values are integer-like (e.g., 1.0, 2.0)
            if np.allclose(X, X.astype(int)):
                X = X.astype(int)
            else:
                raise ValueError("Discrete HMM requires integer observations")
                
        # Check n_features
        if X.shape[1] != self.n_features:
            if self.n_features == 1 and X.shape[1] > 1:
                warnings.warn(f"n_features was 1 but data has {X.shape[1]} features. "
                            f"Updating n_features to {X.shape[1]}")
                self.n_features = X.shape[1]
            else:
                raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
                
        # Infer n_symbols if not set
        if self.n_symbols is None:
            if self.n_features == 1:
                self.n_symbols_ = int(X.max() + 1)
            else:
                self.n_symbols_ = [int(X[:, i].max() + 1) for i in range(self.n_features)]
        else:
            self.n_symbols_ = self.n_symbols
            
        # Validate symbols are in range
        if self.n_features == 1:
            if isinstance(self.n_symbols_, list):
                n_sym = self.n_symbols_[0]
            else:
                n_sym = self.n_symbols_
            if X.min() < 0 or X.max() >= n_sym:
                raise ValueError(f"Symbols must be in range [0, {n_sym-1}]")
        else:
            for i in range(self.n_features):
                n_sym = self.n_symbols_[i] if isinstance(self.n_symbols_, list) else self.n_symbols_
                if X[:, i].min() < 0 or X[:, i].max() >= n_sym:
                    raise ValueError(f"Symbols in feature {i} must be in range [0, {n_sym-1}]")
                    
        # For MultinomialHMM, we need to convert to one-hot encoding if n_features == 1
        if self.n_features == 1:
            # Create the model
            if self.model is None:
                self.model = self._create_model()
                
            # Set initial parameters if provided
            if init_emissionprob is not None:
                self.model.emissionprob_ = init_emissionprob
                self.model.init_params = self.model.init_params.replace('e', '')
            else:
                # Initialize emission probabilities using clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
                kmeans.fit(X)
                
                # Count symbol frequencies in each cluster
                emissionprob = np.zeros((self.n_states, self.n_symbols_))
                for i in range(self.n_states):
                    mask = kmeans.labels_ == i
                    if np.any(mask):
                        for j in range(self.n_symbols_):
                            emissionprob[i, j] = np.sum(X[mask] == j) + 1.0
                        emissionprob[i] /= emissionprob[i].sum()
                    else:
                        # If no samples in cluster, use uniform distribution
                        emissionprob[i] = 1.0 / self.n_symbols_
                
                self.model.emissionprob_ = emissionprob
                self.model.init_params = self.model.init_params.replace('e', '')
                
            if init_startprob is not None:
                self.model.startprob_ = init_startprob
                self.model.init_params = self.model.init_params.replace('s', '')
                
            if init_transmat is not None:
                self.model.transmat_ = init_transmat
                self.model.init_params = self.model.init_params.replace('t', '')
                
            # Call parent fit method
            return super().fit(X, lengths=lengths, **kwargs)
        else:
            # For multiple features, we need a custom implementation
            raise NotImplementedError("Multiple discrete features not yet supported. "
                                    "Consider encoding multiple features into a single feature.")
    
    def _count_model_parameters(self) -> int:
        """Count Discrete-specific parameters."""
        if self.n_symbols_ is None:
            warnings.warn("n_symbols not set, cannot count parameters accurately")
            return 0
            
        n_params = 0
        
        if self.n_features == 1:
            # Emission probabilities (n_states * (n_symbols - 1))
            n_sym = self.n_symbols_ if isinstance(self.n_symbols_, int) else self.n_symbols_[0]
            n_params += self.n_states * (n_sym - 1)
        else:
            # Multiple features
            if isinstance(self.n_symbols_, list):
                for n_sym in self.n_symbols_:
                    n_params += self.n_states * (n_sym - 1)
            else:
                n_params += self.n_features * self.n_states * (self.n_symbols_ - 1)
                
        return n_params
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get Discrete-specific parameters."""
        params = {
            'emission_probabilities': self.model.emissionprob_,
            'n_symbols': self.n_symbols_,
            'n_features': self.n_features
        }
        return params
    
    @property
    def emissionprob_(self) -> np.ndarray:
        """Get the emission probability parameters."""
        self._check_is_fitted()
        return self.model.emissionprob_
    
    def get_state_distribution(self, state: int) -> Dict[str, np.ndarray]:
        """
        Get the emission distribution for a specific state.
        
        Parameters
        ----------
        state : int
            The state index
            
        Returns
        -------
        dist_params : dict
            Dictionary containing 'probabilities' for each symbol in the state
        """
        self._check_is_fitted()
        
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State must be between 0 and {self.n_states-1}")
            
        return {'probabilities': self.emissionprob_[state]}
    
    def decode_symbols(self, symbols: List[str], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert symbolic observations to integer codes for model input.
        
        Parameters
        ----------
        symbols : list of str
            List of unique symbols in order (index corresponds to code)
        X : array-like
            Symbolic observations to encode
            
        Returns
        -------
        X_encoded : array, shape (n_samples, n_features)
            Integer-encoded observations
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Create symbol to code mapping
        symbol_to_code = {symbol: i for i, symbol in enumerate(symbols)}
        
        # Vectorized encoding
        encode_func = np.vectorize(lambda x: symbol_to_code.get(x, -1))
        X_encoded = encode_func(X)
        
        # Check for unknown symbols
        if np.any(X_encoded == -1):
            unknown_mask = X_encoded == -1
            unknown_symbols = np.unique(X[unknown_mask])
            raise ValueError(f"Unknown symbols encountered: {unknown_symbols}")
            
        return X_encoded
    
    def encode_symbols(self, symbols: List[str], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert integer codes back to symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of unique symbols in order (index corresponds to code)
        X : array-like
            Integer codes to convert to symbols
            
        Returns
        -------
        X_decoded : array
            Symbolic observations
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Ensure X contains integers
        X = X.astype(int)
        
        # Check codes are in valid range
        if np.any(X < 0) or np.any(X >= len(symbols)):
            raise ValueError(f"Codes must be in range [0, {len(symbols)-1}]")
            
        # Vectorized decoding
        decode_func = np.vectorize(lambda x: symbols[x])
        return decode_func(X)
    
    def label_switch(self, sort_by: str = 'emission', symbol: int = 0, ascending: bool = True) -> np.ndarray:
        """
        Reorder states based on a criterion to handle label switching.
        
        Parameters
        ----------
        sort_by : str, default='emission'
            Criterion to sort states by. Options:
            - 'emission': Sort by emission probability of a specific symbol
            - 'entropy': Sort by entropy of emission distribution
        symbol : int, default=0
            Symbol to use when sort_by='emission'
        ascending : bool, default=True
            Whether to sort in ascending order
            
        Returns
        -------
        new_order : array, shape (n_states,)
            The new ordering of states
        """
        self._check_is_fitted()
        
        if sort_by == 'emission':
            # Sort by emission probability of specific symbol
            values = self.emissionprob_[:, symbol]
        elif sort_by == 'entropy':
            # Sort by entropy of emission distribution
            values = -np.sum(self.emissionprob_ * np.log(self.emissionprob_ + 1e-10), axis=1)
        else:
            raise ValueError(f"Unknown sort_by criterion: {sort_by}")
            
        if ascending:
            new_order = np.argsort(values)
        else:
            new_order = np.argsort(-values)
            
        # Only reorder if the order actually changes
        if not np.array_equal(new_order, np.arange(self.n_states)):
            # Reorder model parameters
            self.model.emissionprob_ = self.model.emissionprob_[new_order]
            
            # Reorder transition matrix
            self.model.transmat_ = self.model.transmat_[new_order][:, new_order]
            
            # Reorder initial probabilities
            self.model.startprob_ = self.model.startprob_[new_order]
        
        return new_order
    
    def most_likely_symbols(self) -> np.ndarray:
        """
        Get the most likely symbol for each state.
        
        Returns
        -------
        symbols : array, shape (n_states,)
            Most likely symbol for each state
        """
        self._check_is_fitted()
        return np.argmax(self.emissionprob_, axis=1)