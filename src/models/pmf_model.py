# src/models/pmf_model.py
from surprise import SVD, NMF
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any
from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)

class PMFModel(BaseRecommenderModel):
    """Probabilistic Matrix Factorization with optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config['models']['pmf']
        
        # Choose algorithm variant
        if self.config.get('algorithm', 'svd') == 'nmf':
            # Non-negative Matrix Factorization variant
            self.model = NMF(
                n_factors=self.config['n_factors'],
                n_epochs=self.config['n_epochs'],
                biased=self.config['biased'],
                reg_pu=self.config.get('reg_pu', self.config['regularization']),
                reg_qi=self.config.get('reg_qi', self.config['regularization']),
                random_state=self.config['random_state'],
                verbose=self.config.get('verbose', True)
            )
        else:
            # Standard SVD (PMF when biased=False)
            self.model = SVD(
                n_factors=self.config['n_factors'],
                n_epochs=self.config['n_epochs'],
                biased=self.config['biased'],  # False for pure PMF
                lr_all=self.config['learning_rate'],
                reg_all=self.config['regularization'],
                random_state=self.config['random_state'],
                verbose=self.config.get('verbose', True)
            )
        self.trainset = None
        
    def fit(self, trainset):
        """Train PMF model with timing."""
        logger.info("Training PMF model...")
        logger.info(f"Configuration: {self.config}")
        
        start_time = time.time()
        self.trainset = trainset
        self.model.fit(trainset)
        self.training_time = time.time() - start_time
        
        # Store model info
        self.model_info = {
            'n_users': trainset.n_users,
            'n_items': trainset.n_items,
            'n_ratings': trainset.n_ratings,
            'global_mean': trainset.global_mean,
            'training_time': self.training_time
        }
        
        logger.info(f"PMF training completed in {self.training_time:.2f} seconds")
        logger.info(f"Model size: Users {self.model.pu.shape}, Items {self.model.qi.shape}")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair."""
        return self.model.predict(user_id, item_id).est
    
    def predict_batch(self, user_item_pairs: List[Tuple[int, int]]) -> List[float]:
        """Predict ratings for multiple user-item pairs efficiently."""
        predictions = []
        for user_id, item_id in user_item_pairs:
            pred = self.model.predict(user_id, item_id)
            predictions.append(pred.est)
        return predictions
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True,
                  candidate_items: List[int] = None) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        # Get user's inner id
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            logger.warning(f"User {user_id} not in training set")
            return []
        
        # Get candidate items
        if candidate_items is None:
            all_items = self.trainset.all_items()
        else:
            all_items = [self.trainset.to_inner_iid(i) for i in candidate_items 
                        if i in self.trainset._raw2inner_id_items]
        
        # Get items the user has already rated
        seen_items = set()
        if exclude_seen:
            seen_items = {iid for iid, _ in self.trainset.ur[user_inner_id]}
        
        # Predict ratings for all unseen items
        predictions = []
        for item_inner_id in all_items:
            if item_inner_id not in seen_items:
                item_id = self.trainset.to_raw_iid(item_inner_id)
                pred = self.model.predict(user_id, item_id)
                predictions.append((item_id, pred.est))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_items]
    
    def get_user_factors(self) -> np.ndarray:
        """Get user latent factors."""
        return self.model.pu
    
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        return self.model.qi
    
    def get_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get user and item biases if model uses them."""
        if self.config['biased']:
            return self.model.bu, self.model.bi
        else:
            return None, None