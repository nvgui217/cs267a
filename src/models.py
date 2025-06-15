from surprise import SVD
from implicit.bpr import BayesianPersonalizedRanking
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class PMFModel:
    """Probabilistic Matrix Factorization using Surprise SVD."""
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, 
                 lr_all: float = 0.005, reg_all: float = 0.02,
                 random_state: int = 42):
        """Initialize PMF model.
        
        Note: SVD with biased=False is equivalent to PMF.
        """
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            biased=False,  # This makes it PMF
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
            verbose=True
        )
        self.trainset = None
        
    def fit(self, trainset):
        """Train PMF model."""
        logger.info("Training PMF model...")
        self.trainset = trainset
        self.model.fit(trainset)
        logger.info("PMF training completed")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair."""
        return self.model.predict(user_id, item_id).est
    
    def test(self, testset):
        """Test model on testset."""
        return self.model.test(testset)
    
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        return self.model.qi


class BPRModel:
    """Bayesian Personalized Ranking Matrix Factorization."""
    
    def __init__(self, factors: int = 50, iterations: int = 100,
                 learning_rate: float = 0.01, regularization: float = 0.01,
                 random_state: int = 42):
        """Initialize BPR-MF model."""
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            num_threads=0,  # Use all available
            random_state=random_state
        )
        self.user_item_matrix = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.user_to_idx = None
        
    def fit(self, implicit_data: Dict[str, Any]):
        """Train BPR model."""
        logger.info("Training BPR-MF model...")
        
        self.user_item_matrix = implicit_data['train_matrix']
        self.idx_to_user = implicit_data['idx_to_user']
        self.idx_to_item = implicit_data['idx_to_item']
        self.user_to_idx = implicit_data['user_to_idx']
        
        # Train model
        self.model.fit(self.user_item_matrix, show_progress=True)
        logger.info("BPR-MF training completed")
    
    def predict_for_user(self, user_id: int, item_ids: List[int]) -> List[float]:
        """Predict scores for a user and multiple items."""
        if user_id not in self.user_to_idx:
            return [0.0] * len(item_ids)
        
        user_idx = self.user_to_idx[user_id]
        scores = []
        
        for item_id in item_ids:
            if item_id in self.user_to_idx:
                item_idx = self.user_to_idx[item_id]
                score = np.dot(
                    self.model.user_factors[user_idx],
                    self.model.item_factors[item_idx]
                )
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return scores
    
    def recommend(self, user_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get recommendations
        recommendations, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n_items,
            filter_already_liked_items=True
        )
        
        # Convert indices to IDs
        results = []
        for item_idx, score in zip(recommendations, scores):
            item_id = self.idx_to_item[item_idx]
            results.append((item_id, float(score)))
        
        return results
    
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        return self.model.item_factors