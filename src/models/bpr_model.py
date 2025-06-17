from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
import numpy as np
import scipy.sparse as sp
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from .base_model import BaseRecommenderModel

logger = logging.getLogger(__name__)

class BPRModel(BaseRecommenderModel):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config['models']['bpr']
        
        algorithm = self.config.get('algorithm', 'bpr')
        
        # Use float32 for memory efficiency
        dtype = np.float32 if self.config['dtype'] == 'float32' else np.float64
        
        if algorithm == 'als':
            self.model = AlternatingLeastSquares(
                factors=self.config['factors'],
                iterations=self.config['iterations'],
                regularization=self.config['regularization'],
                num_threads=self.config['num_threads'],
                random_state=self.config['random_state'],
                dtype=dtype
            )
        elif algorithm == 'lmf':
            self.model = LogisticMatrixFactorization(
                factors=self.config['factors'],
                iterations=self.config['iterations'],
                learning_rate=self.config['learning_rate'],
                regularization=self.config['regularization'],
                num_threads=self.config['num_threads'],
                random_state=self.config['random_state'],
                dtype=dtype
            )
        else:
            self.model = BayesianPersonalizedRanking(
                factors=self.config['factors'],
                iterations=self.config['iterations'],
                learning_rate=self.config['learning_rate'],
                regularization=self.config['regularization'],
                num_threads=self.config['num_threads'],
                random_state=self.config['random_state'],
                dtype=dtype,
                verify_negative_samples=self.config.get('verify_negative_samples', True)
            )
        
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.user_to_idx = None
        self.item_to_idx = None
        
    def fit(self, train_data: Dict[str, Any]):
        logger.info(f"Training BPR")
        logger.info(f"Configuration: {self.config}")
        
        self.user_item_matrix = train_data['train_matrix']
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        self.idx_to_user = train_data['idx_to_user']
        self.idx_to_item = train_data['idx_to_item']
        self.user_to_idx = train_data['user_to_idx']
        self.item_to_idx = train_data['item_to_idx']
        
        # Train model
        start_time = time.time()
        self.model.fit(self.user_item_matrix, show_progress=True)
        self.training_time = time.time() - start_time
        
        # Store model
        self.model_info = {
            'n_users': self.user_item_matrix.shape[0],
            'n_items': self.user_item_matrix.shape[1],
            'n_interactions': self.user_item_matrix.nnz,
            'density': self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * 
                                                   self.user_item_matrix.shape[1]),
            'training_time': self.training_time,
            'algorithm': self.config.get('algorithm', 'bpr')
        }
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        logger.info(f"Model size: Users {self.model.user_factors.shape}, "
                   f"Items {self.model.item_factors.shape}")
    
    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_to_idx:
            logger.debug(f"User {user_id} not in training set")
            return 0.0
            
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx.get(item_id, -1)
        
        if item_idx == -1:
            logger.debug(f"Item {item_id} not in training set")
            return 0.0
        
        score = np.dot(
            self.model.user_factors[user_idx],
            self.model.item_factors[item_idx]
        )
        return float(score)
    
    def predict_batch(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        if user_id not in self.user_to_idx:
            return np.zeros(len(item_ids))
        
        user_idx = self.user_to_idx[user_id]
        user_factor = self.model.user_factors[user_idx]
        
        scores = []
        for item_id in item_ids:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                score = np.dot(user_factor, self.model.item_factors[item_idx])
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def recommend(self, user_id: int, n_items: int = 10,
                  filter_already_liked_items: bool = True,
                  items_to_recommend: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not in training set")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        recommendations, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n_items,
            filter_already_liked_items=filter_already_liked_items,
            items=items_to_recommend
        )
        
        # Convert indices back to IDs
        results = []
        for item_idx, score in zip(recommendations, scores):
            item_id = self.idx_to_item[item_idx]
            results.append((item_id, float(score)))
        
        return results
    
    def recommend_batch(self, user_ids: List[int], n_items: int = 10,
                       filter_already_liked_items: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        # Convert user IDs to indices
        user_indices = []
        valid_user_ids = []
        for user_id in user_ids:
            if user_id in self.user_to_idx:
                user_indices.append(self.user_to_idx[user_id])
                valid_user_ids.append(user_id)
        
        if not user_indices:
            return {}
        
        # Get batch recommendations
        recommendations = self.model.recommend_all(
            user_items=self.user_item_matrix[user_indices],
            N=n_items,
            filter_already_liked_items=filter_already_liked_items
        )
        
        results = {}
        for i, user_id in enumerate(valid_user_ids):
            user_recs = []
            for j in range(n_items):
                if j < len(recommendations[i]):
                    item_idx = recommendations[i][j]
                    item_id = self.idx_to_item[item_idx]
                    score = np.dot(
                        self.model.user_factors[user_indices[i]],
                        self.model.item_factors[item_idx]
                    )
                    user_recs.append((item_id, float(score)))
            results[user_id] = user_recs
        
        return results
    
    def get_user_factors(self) -> np.ndarray:
        return self.model.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        return self.model.item_factors
    
    def explain_recommendations(self, user_id: int, item_ids: List[int],
                              n_top_factors: int = 5) -> Dict[int, Dict[str, Any]]:
        if user_id not in self.user_to_idx:
            return {}
        
        user_idx = self.user_to_idx[user_id]
        user_factor = self.model.user_factors[user_idx]
        
        explanations = {}
        for item_id in item_ids:
            if item_id not in self.item_to_idx:
                continue
                
            item_idx = self.item_to_idx[item_id]
            item_factor = self.model.item_factors[item_idx]
            
            # Calculate factor contributions
            factor_contributions = user_factor * item_factor
            top_factor_indices = np.argsort(np.abs(factor_contributions))[::-1][:n_top_factors]
            
            explanations[item_id] = {
                'total_score': float(np.sum(factor_contributions)),
                'top_factors': [
                    {
                        'factor_id': int(idx),
                        'contribution': float(factor_contributions[idx]),
                        'user_value': float(user_factor[idx]),
                        'item_value': float(item_factor[idx])
                    }
                    for idx in top_factor_indices
                ]
            }
        
        return explanations