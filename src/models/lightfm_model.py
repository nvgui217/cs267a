# src/models/lightfm_model.py
from lightfm import LightFM
import numpy as np
import scipy.sparse as sp
import logging
import time
from typing import List, Tuple, Dict, Any
from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)

class LightFMModel(BaseRecommenderModel):
    """LightFM model for hybrid recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config['models']['lightfm']
        
        self.model = LightFM(
            no_components=self.config['no_components'],
            learning_rate=self.config['learning_rate'],
            loss=self.config['loss'],
            max_sampled=self.config['max_sampled'],
            item_alpha=self.config['item_alpha'],
            user_alpha=self.config['user_alpha'],
            random_state=self.config['random_state']
        )
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.idx_to_user = None  # <-- Initialize attribute
        self.item_features = None
        self.user_item_matrix = None

    def fit(self, train_data: Dict[str, Any]):
        """Train LightFM model."""
        logger.info("Training LightFM model...")
        logger.info(f"Configuration: {self.config}")
        
        # Store mappings and matrices needed for prediction/recommendation
        self.user_to_idx = train_data['user_to_idx']
        self.item_to_idx = train_data['item_to_idx']
        self.idx_to_item = train_data['idx_to_item']
        self.idx_to_user = train_data['idx_to_user'] # <-- ADDED THIS LINE
        self.item_features = train_data['item_features']
        self.user_item_matrix = train_data['train_matrix']
        
        start_time = time.time()
        
        self.model.fit(
            self.user_item_matrix,
            item_features=self.item_features,
            epochs=self.config['epochs'],
            num_threads=self.config['num_threads'],
            verbose=True
        )
        
        self.training_time = time.time() - start_time
        
        self.model_info = {
            'training_time': self.training_time,
            'algorithm': 'LightFM',
            'loss': self.config['loss']
        }
        logger.info(f"LightFM training completed in {self.training_time:.2f} seconds")

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict preference score."""
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return 0.0
        
        scores = self.model.predict(
            np.array([user_idx]), 
            np.array([item_idx]),
            item_features=self.item_features,
            num_threads=self.config['num_threads']
        )
        return float(scores[0])

    def recommend(self, user_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """Get top-N recommendations."""
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return []
            
        n_users, n_all_items = self.user_item_matrix.shape
        item_indices = np.arange(n_all_items)
        
        scores = self.model.predict(
            user_idx,
            item_indices,
            item_features=self.item_features,
            num_threads=self.config['num_threads']
        )
        
        # Filter out items the user has already seen
        known_positives = self.user_item_matrix.getrow(user_idx).indices
        scores[known_positives] = -np.inf

        top_indices = np.argsort(-scores)[:n_items]
        
        recommendations = [
            (self.idx_to_item[item_idx], float(scores[item_idx])) for item_idx in top_indices
        ]
        return recommendations
    
    def get_user_factors(self) -> np.ndarray:
        """Get user latent factors."""
        biases, factors = self.model.get_user_representations()
        return factors

    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        biases, factors = self.model.get_item_representations(features=self.item_features)
        return factors