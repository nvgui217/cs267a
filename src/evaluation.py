import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluation metrics for recommender systems."""
    
    @staticmethod
    def compute_rmse(predictions: List) -> float:
        """Compute RMSE for explicit feedback predictions."""
        true_ratings = [true_r for (_, _, true_r, est, _) in predictions]
        estimated_ratings = [est for (_, _, true_r, est, _) in predictions]
        return np.sqrt(mean_squared_error(true_ratings, estimated_ratings))
    
    @staticmethod
    def compute_mae(predictions: List) -> float:
        """Compute MAE for explicit feedback predictions."""
        true_ratings = [true_r for (_, _, true_r, est, _) in predictions]
        estimated_ratings = [est for (_, _, true_r, est, _) in predictions]
        return np.mean(np.abs(np.array(true_ratings) - np.array(estimated_ratings)))
    
    @staticmethod
    def compute_precision_at_k(recommended_items: List[List[int]], 
                              relevant_items: List[List[int]], 
                              k: int = 10) -> float:
        """Compute Precision@K for implicit feedback."""
        precisions = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            # Take top-k recommendations
            rec_k = set(rec_items[:k])
            rel_set = set(rel_items)
            
            # Calculate precision
            if len(rec_k) > 0:
                precision = len(rec_k & rel_set) / len(rec_k)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def evaluate_pmf_as_ranker(pmf_model, test_df, train_df, k: int = 10) -> float:
        """Evaluate PMF model as a ranker for comparison with BPR."""
        # Get unique users in test set
        test_users = test_df['user_id'].unique()
        
        # Get all items
        all_items = set(train_df['item_id'].unique()) | set(test_df['item_id'].unique())
        
        precisions = []
        
        for user_id in test_users[:100]:  # Sample for efficiency
            # Get items user rated highly in test set (>= 4)
            relevant_items = test_df[
                (test_df['user_id'] == user_id) & 
                (test_df['rating'] >= 4)
            ]['item_id'].tolist()
            
            if not relevant_items:
                continue
            
            # Get items user hasn't seen in training
            seen_items = set(train_df[train_df['user_id'] == user_id]['item_id'].tolist())
            candidate_items = list(all_items - seen_items)
            
            if not candidate_items:
                continue
            
            # Predict scores for all candidate items
            predictions = []
            for item_id in candidate_items:
                try:
                    score = pmf_model.predict(user_id, item_id)
                    predictions.append((item_id, score))
                except:
                    continue
            
            # Sort by score and get top-k
            predictions.sort(key=lambda x: x[1], reverse=True)
            recommended = [item_id for item_id, _ in predictions[:k]]
            
            # Calculate precision
            if recommended:
                precision = len(set(recommended) & set(relevant_items)) / len(recommended)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0