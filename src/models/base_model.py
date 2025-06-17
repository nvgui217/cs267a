# src/models/base_model.py
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import logging
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseRecommenderModel(ABC):
    """Abstract base class for recommender models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.training_time = None
        self.model_info = {}
    
    @abstractmethod
    def fit(self, train_data):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Make a prediction for a user-item pair."""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        pass
    
    @abstractmethod
    def get_user_factors(self) -> np.ndarray:
        """Get user latent factors."""
        pass
    
    @abstractmethod
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        pass
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar users based on factor similarity."""
        user_factors = self.get_user_factors()
        
        # Get user index
        if hasattr(self, 'trainset'):
            # For surprise models
            try:
                user_idx = self.trainset.to_inner_uid(user_id)
            except ValueError:
                logger.warning(f"User {user_id} not in training set")
                return []
        else:
            # For implicit models
            if user_id not in self.user_to_idx:
                logger.warning(f"User {user_id} not in training set")
                return []
            user_idx = self.user_to_idx[user_id]
        
        # Get user factor
        user_factor = user_factors[user_idx]
        
        # Compute similarities with all other users
        similarities = []
        for other_idx in range(len(user_factors)):
            if other_idx != user_idx:
                other_factor = user_factors[other_idx]
                # Cosine similarity
                sim = np.dot(user_factor, other_factor) / (
                    np.linalg.norm(user_factor) * np.linalg.norm(other_factor) + 1e-8
                )
                
                # Convert back to user ID
                if hasattr(self, 'trainset'):
                    other_id = self.trainset.to_raw_uid(other_idx)
                else:
                    other_id = self.idx_to_user[other_idx]
                    
                similarities.append((other_id, float(sim)))
        
        # Sort and return top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar items based on factor similarity."""
        item_factors = self.get_item_factors()
        
        # Get item index
        if hasattr(self, 'trainset'):
            # For surprise models
            try:
                item_idx = self.trainset.to_inner_iid(item_id)
            except ValueError:
                logger.warning(f"Item {item_id} not in training set")
                return []
        else:
            # For implicit models
            if item_id not in self.item_to_idx:
                logger.warning(f"Item {item_id} not in training set")
                return []
            item_idx = self.item_to_idx[item_id]
        
        # Get item factor
        item_factor = item_factors[item_idx]
        
        # Compute similarities with all other items
        similarities = []
        for other_idx in range(len(item_factors)):
            if other_idx != item_idx:
                other_factor = item_factors[other_idx]
                # Cosine similarity
                sim = np.dot(item_factor, other_factor) / (
                    np.linalg.norm(item_factor) * np.linalg.norm(other_factor) + 1e-8
                )
                
                # Convert back to item ID
                if hasattr(self, 'trainset'):
                    other_id = self.trainset.to_raw_iid(other_idx)
                else:
                    other_id = self.idx_to_item[other_idx]
                    
                similarities.append((other_id, float(sim)))
        
        # Sort and return top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def save_model(self, filepath: str):
        """Save model to disk with metadata."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(filepath)
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        
        # Save metadata
        metadata = {
            'model_type': self.__class__.__name__,
            'config': self.config,
            'training_time': self.training_time,
            'model_info': self.model_info
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata if exists
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model.model_info = metadata.get('model_info', {})
                
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size in terms of parameters."""
        user_factors = self.get_user_factors()
        item_factors = self.get_item_factors()
        
        n_users, n_factors = user_factors.shape
        n_items, _ = item_factors.shape
        
        return {
            'n_users': n_users,
            'n_items': n_items,
            'n_factors': n_factors,
            'total_parameters': n_users * n_factors + n_items * n_factors
        }