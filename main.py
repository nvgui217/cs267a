import logging
import yaml
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any
import pandas as pd 

from src.data.data_loader import MovieLensDataLoader
from src.models import PMFModel, BPRModel, LightFMModel
from src.evaluation.evaluator import ModelEvaluator
from src.optimization.hyperparameter_tuning import HyperparameterTuner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender_system.log', mode='w'), # Overwrite log each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['output']['results_dir'])
    for subdir in ['models', 'metrics', 'plots', 'predictions', 'optimization']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    data_loader = MovieLensDataLoader(config)
    evaluator = ModelEvaluator(config)
    
    # Load and prepare data
    print("Data Loading and Preparation")
    data = data_loader.prepare_data_splits()
    movies_df = data['movies_df']
    
    if config['optimization']['run_optimization']:
        print("Hyperparameter Optimization")
        tuner = HyperparameterTuner(config)
        
        train_split, val_split = data_loader.create_validation_split(data['train_df'])
        
        logger.info("\n Tuning PMF Hyperparameters")
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1, 5))
        pmf_train_data = Dataset.load_from_df(train_split[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
        pmf_val_data = [(row.user_id, row.item_id, row.rating) for row in val_split.itertuples(index=False)]
        
        best_pmf_params = tuner.tune_pmf(pmf_train_data, pmf_val_data)
        
        logger.info(f"PMF config best params: {best_pmf_params}")
        config['models']['pmf'].update(best_pmf_params)

        logger.info("\n Tuning BPR Hyperparameters ")
        bpr_val_data = data_loader._prepare_implicit_data(
            data['ratings_df'], train_split, val_split, 
            threshold=config['data']['min_rating_threshold']
        )
        best_bpr_params = tuner.tune_bpr(bpr_val_data, bpr_val_data)
        
        logger.info(f"BPR config best params: {best_bpr_params}")
        config['models']['bpr'].update(best_bpr_params)
        
    else:
        pass

    #Train final models
    print("Training Final Models")
    
    logger.info(f"Final PMF Configuration: {config['models']['pmf']}")
    pmf_model = PMFModel(config)
    pmf_model.fit(data['explicit']['trainset'])
    pmf_size = pmf_model.get_model_size()
    
    logger.info(f"Final BPR Configuration: {config['models']['bpr']}")
    bpr_model = BPRModel(config)
    bpr_model.fit(data['implicit'])
    bpr_size = bpr_model.get_model_size()

    lightfm_model = None
    if config.get('optimization', {}).get('run_lightfm', False):
        logger.info(f"Final LightFM Configuration: {config['models']['lightfm']}")
        lightfm_model = LightFMModel(config)
        lightfm_model.fit(data['implicit'])
        lightfm_size = lightfm_model.get_model_size()
    
    # Evaluate final models
    all_metrics = {'experiment_info': {'config': config}}
    
    logger.info("\n Evaluating PMF Model")
    pmf_results = evaluator.evaluate_pmf(pmf_model, data['explicit']['testset'])
    all_metrics['pmf_results'] = {**pmf_results, 'model_info': pmf_model.model_info, 'model_size': pmf_size}
    
    logger.info("\n Evaluating BPR-MF Model")
    bpr_results = evaluator.evaluate_bpr(bpr_model, data['implicit']['test_matrix'], data['implicit']['train_matrix'])
    all_metrics['bpr_results'] = {**bpr_results, 'model_info': bpr_model.model_info, 'model_size': bpr_size}

    if lightfm_model:
        logger.info("\n Evaluating LightFM Model")
        lightfm_results = evaluator.evaluate_bpr(lightfm_model, data['implicit']['test_matrix'], data['implicit']['train_matrix'])
        all_metrics['lightfm_results'] = {**lightfm_results, 'model_info': lightfm_model.model_info, 'model_size': lightfm_size}

    elapsed_time = time.time() - start_time
    all_metrics['experiment_info']['timestamp'] = datetime.now().isoformat()
    all_metrics['experiment_info']['total_execution_time'] = elapsed_time

    metrics_path = results_dir / 'metrics' / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("COMPLETE")
    logger.info(f"Final PMF RMSE: {pmf_results.get('rmse', 'N/A'):.4f}")
    logger.info(f"Final BPR Precision@10: {bpr_results.get('precision@10', 'N/A'):.4f}")
    if lightfm_model:
        logger.info(f"Final LightFM Precision@10: {all_metrics.get('lightfm_results', {}).get('precision@10', 'N/A'):.4f}")

if __name__ == "__main__":
    main()