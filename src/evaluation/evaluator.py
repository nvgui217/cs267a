import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import time
from sklearn.model_selection import KFold
from .metrics import RecommenderMetrics

logger = logging.getLogger(__name__)

class ModelEvaluator:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = RecommenderMetrics()
        
    def evaluate_pmf(self, model, test_data: List[Tuple],
                    calc_confidence: bool = False) -> Dict[str, float]:
        logger.info("Evaluating PMF model")
        
        start_time = time.time()
        
        predictions = []
        confidence_intervals = []
        
        for user_id, item_id, true_rating in tqdm(test_data, desc="Making predictions"):
            pred = model.predict(user_id, item_id)
            predictions.append((true_rating, pred))
            
            if calc_confidence and hasattr(model, 'predict_with_confidence'):
                lower, upper = model.predict_with_confidence(user_id, item_id)
                confidence_intervals.append((lower, upper))
        
        results = {
            'rmse': self.metrics.rmse(predictions),
            'mae': self.metrics.mae(predictions),
            'mse': self.metrics.mse(predictions),
            'n_predictions': len(predictions),
            'evaluation_time': time.time() - start_time
        }
        
        if confidence_intervals:
            coverage = sum(1 for (true_r, _), (lower, upper) in 
                         zip(predictions, confidence_intervals)
                         if lower <= true_r <= upper) / len(predictions)
            results['confidence_coverage'] = coverage
            results['avg_interval_width'] = np.mean([upper - lower 
                                                    for lower, upper in confidence_intervals])
        
        logger.info(f"PMF Results RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        return results
    
    def evaluate_bpr(self, model, test_matrix, train_matrix, 
                    k_values: List[int] = None,
                    calc_diversity: bool = True) -> Dict[str, float]:
        logger.info("Evaluating BPR-MF model")
        
        if k_values is None:
            k_values = self.config['evaluation']['k_values']
        
        start_time = time.time()
        
        test_csr = test_matrix.tocsr()
        test_users = np.where(np.diff(test_csr.indptr) > 0)[0]
        
        max_eval_users = self.config['evaluation'].get('max_eval_users', 1000)
        if len(test_users) > max_eval_users:
            test_users = np.random.choice(test_users, max_eval_users, replace=False)
        
        all_recommendations = []
        all_ground_truth = []
        
        batch_size = 100
        for i in tqdm(range(0, len(test_users), batch_size), desc="Generating recommendations"):
            batch_users = test_users[i:i+batch_size]
            batch_user_ids = [model.idx_to_user[idx] for idx in batch_users]
            
            if hasattr(model, 'recommend_batch'):
                batch_recs = model.recommend_batch(batch_user_ids, n_items=max(k_values))
                
                for user_idx, user_id in zip(batch_users, batch_user_ids):
                    if user_id in batch_recs:
                        rec_items = [item_id for item_id, _ in batch_recs[user_id]]
                        true_items = test_csr[user_idx].indices.tolist()
                        
                        if true_items:
                            all_recommendations.append(rec_items)
                            all_ground_truth.append(true_items)
            else:
                for user_idx, user_id in zip(batch_users, batch_user_ids):
                    true_items = test_csr[user_idx].indices.tolist()
                    if not true_items:
                        continue
                    
                    recommendations = model.recommend(user_id, n_items=max(k_values))
                    rec_items = [item_id for item_id, _ in recommendations]
                    
                    all_recommendations.append(rec_items)
                    all_ground_truth.append(true_items)
        
        results = {}
        for k in k_values:
            results[f'precision@{k}'] = self.metrics.precision_at_k(
                all_recommendations, all_ground_truth, k
            )
            results[f'recall@{k}'] = self.metrics.recall_at_k(
                all_recommendations, all_ground_truth, k
            )
            results[f'f1@{k}'] = self.metrics.f1_at_k(
                all_recommendations, all_ground_truth, k
            )
            results[f'ndcg@{k}'] = self.metrics.ndcg_at_k(
                all_recommendations, all_ground_truth, k
            )
            results[f'map@{k}'] = self.metrics.map_at_k(
                all_recommendations, all_ground_truth, k
            )
            results[f'mrr@{k}'] = self.metrics.mrr_at_k(
                all_recommendations, all_ground_truth, k
            )
        
        all_items = set(range(test_matrix.shape[1]))
        results['coverage'] = self.metrics.coverage(all_recommendations, all_items)
        
        if calc_diversity and hasattr(model, 'get_item_factors'):
            item_factors = model.get_item_factors()
            results['diversity'] = self.metrics.diversity(all_recommendations, item_factors)
        
        train_csr = train_matrix.tocsr()
        item_popularity = {}
        for item_idx in range(train_matrix.shape[1]):
            item_popularity[item_idx] = train_csr[:, item_idx].nnz / train_matrix.shape[0]
        
        results['novelty'] = self.metrics.novelty(all_recommendations, item_popularity)
        
        results['evaluation_time'] = time.time() - start_time
        results['n_users_evaluated'] = len(all_recommendations)
        
        logger.info(f"BPR Results - P@10: {results.get('precision@10', 0):.4f}, "
                   f"R@10: {results.get('recall@10', 0):.4f}, "
                   f"NDCG@10: {results.get('ndcg@10', 0):.4f}, "
                   f"Coverage: {results['coverage']:.4f}")
        
        return results
    
    def cross_validate_pmf(self, model_class, data, n_folds: int = 5,
                          **model_kwargs) -> Dict[str, List[float]]:
        ratings_array = np.array([(u, i, r) for u, i, r in data['explicit']['trainset'].all_ratings()])
        
        kf = KFold(n_splits=n_folds, shuffle=True, 
                  random_state=self.config['models']['pmf']['random_state'])
        
        fold_results = {
            'rmse': [],
            'mae': [],
            'fold_times': []
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(ratings_array)):
            logger.info(f"Processing fold {fold_idx + 1}/{n_folds}")
            
            fold_start = time.time()
            
            # Create train and validation sets
            train_ratings = ratings_array[train_idx]
            val_ratings = ratings_array[val_idx]
            
            from surprise import Dataset, Reader
            reader = Reader(rating_scale=(1, 5))
            
            train_df = pd.DataFrame(train_ratings, columns=['user_id', 'item_id', 'rating'])
            train_data = Dataset.load_from_df(train_df, reader)
            trainset = train_data.build_full_trainset()
            
            model = model_class(self.config)
            model.fit(trainset)
            
            # Evaluate on validation
            val_predictions = []
            for user_id, item_id, true_rating in val_ratings:
                pred = model.predict(int(user_id), int(item_id))
                val_predictions.append((float(true_rating), pred))
            
            fold_results['rmse'].append(self.metrics.rmse(val_predictions))
            fold_results['mae'].append(self.metrics.mae(val_predictions))
            fold_results['fold_times'].append(time.time() - fold_start)
        
        for metric in ['rmse', 'mae']:
            values = fold_results[metric]
            fold_results[f'{metric}_mean'] = np.mean(values)
            fold_results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"RMSE: {fold_results['rmse_mean']:.4f} "
                   f"(with {fold_results['rmse_std']:.4f})")
        
        return fold_results
    
    def evaluate_time_based_split(self, model, test_data: pd.DataFrame,
                                 time_windows: int = 5) -> Dict[str, List[float]]:
        test_data_sorted = test_data.sort_values('timestamp')
        window_size = len(test_data_sorted) // time_windows
        
        results = {
            'window_rmse': [],
            'window_sizes': [],
            'window_timestamps': []
        }
        
        for i in range(time_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < time_windows - 1 else len(test_data_sorted)
            
            window_data = test_data_sorted.iloc[start_idx:end_idx]
            
            predictions = []
            for _, row in window_data.iterrows():
                pred = model.predict(row['user_id'], row['item_id'])
                predictions.append((row['rating'], pred))
            
            if predictions:
                rmse = self.metrics.rmse(predictions)
                results['window_rmse'].append(rmse)
                results['window_sizes'].append(len(predictions))
                results['window_timestamps'].append(
                    window_data['timestamp'].mean()
                )
        
        return results