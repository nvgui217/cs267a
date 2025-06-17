from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import logging
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseRecommenderModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.training_time = None
        self.model_info = {}
    
    @abstractmethod
    def fit(self, train_data):
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        pass
    
    @abstractmethod
    def get_user_factors(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_item_factors(self) -> np.ndarray:
        pass
    
    def get_model_size(self) -> Dict[str, int]:
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