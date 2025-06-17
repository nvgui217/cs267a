import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RecommenderMetrics:
    def rmse(self, predictions: List[Tuple[float, float]]) -> float:
        true_ratings = [true for true, pred in predictions]
        pred_ratings = [pred for true, pred in predictions]
        return np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    
    def mae(self, predictions: List[Tuple[float, float]]) -> float:
        true_ratings = [true for true, pred in predictions]
        pred_ratings = [pred for true, pred in predictions]
        return mean_absolute_error(true_ratings, pred_ratings)
    
    def mse(self, predictions: List[Tuple[float, float]]) -> float:
        true_ratings = [true for true, pred in predictions]
        pred_ratings = [pred for true, pred in predictions]
        return mean_squared_error(true_ratings, pred_ratings)
    
    def precision_at_k(self, recommended_items: List[List[int]], 
                      relevant_items: List[List[int]], 
                      k: int) -> float:
        precisions = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            rec_k = set(rec_items[:k])
            rel_set = set(rel_items)
            
            if len(rec_k) > 0:
                precision = len(rec_k & rel_set) / len(rec_k)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self, recommended_items: List[List[int]], 
                   relevant_items: List[List[int]], 
                   k: int) -> float:
        """Calculate Recall@K."""
        recalls = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            rec_k = set(rec_items[:k])
            rel_set = set(rel_items)
            
            # Calculate recall
            if len(rel_set) > 0:
                recall = len(rec_k & rel_set) / len(rel_set)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def f1_at_k(self, recommended_items: List[List[int]], 
                relevant_items: List[List[int]], 
                k: int) -> float:
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended_items: List[List[int]], 
                  relevant_items: List[List[int]], 
                  k: int,
                  relevance_scores: Optional[List[Dict[int, float]]] = None) -> float:
        ndcgs = []
        
        for i, (rec_items, rel_items) in enumerate(zip(recommended_items, relevant_items)):
            if relevance_scores and i < len(relevance_scores):
                rel_scores = relevance_scores[i]
            else:
                rel_scores = {item: 1.0 for item in rel_items}
            
            # DCG
            dcg = 0.0
            for j, item in enumerate(rec_items[:k]):
                if item in rel_scores:
                    # Position starts at 1, not 0
                    dcg += rel_scores[item] / np.log2(j + 2)
            
            ideal_scores = sorted(rel_scores.values(), reverse=True)[:k]
            idcg = sum(score / np.log2(j + 2) for j, score in enumerate(ideal_scores))
            
            # NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def map_at_k(self, recommended_items: List[List[int]], 
                 relevant_items: List[List[int]], 
                 k: int) -> float:
        aps = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            rel_set = set(rel_items)
            
            if not rel_set:
                continue
            
            hits = 0
            sum_precision = 0.0
            
            for j, item in enumerate(rec_items[:k]):
                if item in rel_set:
                    hits += 1
                    sum_precision += hits / (j + 1)
            
            if hits > 0:
                ap = sum_precision / min(len(rel_set), k)
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def mrr_at_k(self, recommended_items: List[List[int]], 
                 relevant_items: List[List[int]], 
                 k: int) -> float:
        rrs = []
        
        for rec_items, rel_items in zip(recommended_items, relevant_items):
            rel_set = set(rel_items)
            
            for j, item in enumerate(rec_items[:k]):
                if item in rel_set:
                    rrs.append(1.0 / (j + 1))
                    break
            else:
                rrs.append(0.0)
        
        return np.mean(rrs) if rrs else 0.0
    
    def coverage(self, recommended_items: List[List[int]], 
                 all_items: set,
                 k: Optional[int] = None) -> float:
        recommended_unique = set()
        for rec_items in recommended_items:
            if k:
                recommended_unique.update(rec_items[:k])
            else:
                recommended_unique.update(rec_items)
        
        return len(recommended_unique) / len(all_items) if all_items else 0.0
    
    def diversity(self, recommended_items: List[List[int]], 
                  item_features: np.ndarray,
                  k: Optional[int] = None) -> float:
        diversities = []
        
        for rec_items in recommended_items:
            if k:
                rec_items = rec_items[:k]
                
            if len(rec_items) < 2:
                continue
                
            # Calculate pairwise distances
            distances = []
            for i in range(len(rec_items)):
                for j in range(i + 1, len(rec_items)):
                    if rec_items[i] < len(item_features) and rec_items[j] < len(item_features):
                        feat_i = item_features[rec_items[i]]
                        feat_j = item_features[rec_items[j]]
                        
                        cos_sim = np.dot(feat_i, feat_j) / (
                            np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8
                        )
                        distances.append(1 - cos_sim)
            
            if distances:
                diversities.append(np.mean(distances))
        
        return np.mean(diversities) if diversities else 0.0
    
    def novelty(self, recommended_items: List[List[int]], 
                item_popularity: Dict[int, float],
                k: Optional[int] = None) -> float:
        novelties = []
        
        for rec_items in recommended_items:
            if k:
                rec_items = rec_items[:k]
            
            item_novelties = []
            for item in rec_items:
                if item in item_popularity:
                    novelty = -np.log2(item_popularity[item] + 1e-10)
                    item_novelties.append(novelty)
            
            if item_novelties:
                novelties.append(np.mean(item_novelties))
        
        return np.mean(novelties) if novelties else 0.0
    
    def serendipity(self, recommended_items: List[List[int]], 
                    relevant_items: List[List[int]],
                    expected_items: List[List[int]],
                    k: int) -> float:
        serendipities = []
        
        for rec_items, rel_items, exp_items in zip(recommended_items, relevant_items, expected_items):
            rec_k = set(rec_items[:k])
            rel_set = set(rel_items)
            exp_set = set(exp_items[:k])
            
            serendipitous = (rec_k & rel_set) - exp_set
            
            if len(rec_k) > 0:
                serendipity = len(serendipitous) / len(rec_k)
                serendipities.append(serendipity)
        
        return np.mean(serendipities) if serendipities else 0.0