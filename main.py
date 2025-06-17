# main.py
import logging
import yaml
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
from typing import Dict, Any
import pandas as pd 

# Import all modules
from src.data.data_loader import MovieLensDataLoader
from src.models import PMFModel, BPRModel, LightFMModel
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.plots import RecommenderVisualizer
from src.optimization.hyperparameter_tuning import HyperparameterTuner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_separator(title: str = "", char: str = "=", length: int = 80):
    """Print a formatted separator line."""
    if title:
        padding = (length - len(title) - 2) // 2
        logger.info(f"\n{char * padding} {title} {char * padding}")
    else:
        logger.info(char * length)

def main():
    """Main execution function with all features."""
    start_time = time.time()
    
    print_separator("CS 267A - MovieLens Recommender System Implementation")
    logger.info("Baseline: PMF with scikit-surprise")
    logger.info("Medium Goal: BPR-MF with implicit library")
    logger.info("Advanced Features: Hyperparameter Tuning, Cross-validation, Ensemble, LightFM")
    print_separator()
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory structure
    results_dir = Path(config['output']['results_dir'])
    for subdir in ['models', 'metrics', 'plots', 'predictions', 'optimization']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_loader = MovieLensDataLoader(config)
    evaluator = ModelEvaluator(config)
    visualizer = RecommenderVisualizer(str(results_dir / 'plots'))
    
    # Step 1: Load and prepare data
    print_separator("STEP 1: Data Loading and Preparation")
    
    data = data_loader.prepare_data_splits()
    movies_df = data['movies_df']
    users_df = data_loader.load_user_info()

    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total ratings: {len(data['ratings_df']):,}")
    logger.info(f"Train ratings: {len(data['train_df']):,}")
    logger.info(f"Test ratings: {len(data['test_df']):,}")
    
    # Step 2: Hyperparameter Tuning (if enabled)
    if config['optimization']['run_optimization']:
        print_separator("STEP 2: Hyperparameter Optimization")
        
        tuner = HyperparameterTuner(config)
        
        train_split, val_split = data_loader.create_validation_split(data['train_df'])
        
        # ... (Tuning logic can be added here) ...
        
    else:
        print_separator("STEP 2: Skipping Hyperparameter Optimization")

    # Step 3: Train models
    print_separator("STEP 3: Training Models")
    
    # Train PMF
    logger.info("\n--- Training PMF Model ---")
    pmf_model = PMFModel(config)
    pmf_model.fit(data['explicit']['trainset'])
    pmf_size = pmf_model.get_model_size()
    logger.info(f"PMF model size: {pmf_size}")
    
    # Train BPR
    logger.info("\n--- Training BPR-MF Model ---")
    bpr_model = BPRModel(config)
    bpr_model.fit(data['implicit'])
    bpr_size = bpr_model.get_model_size()
    logger.info(f"BPR model size: {bpr_size}")

    # Train LightFM if enabled
    lightfm_model = None
    if config.get('optimization', {}).get('run_lightfm', False):
        logger.info("\n--- Training LightFM Model ---")
        lightfm_model = LightFMModel(config)
        lightfm_model.fit(data['implicit']) 
        lightfm_size = lightfm_model.get_model_size()
        logger.info(f"LightFM model size: {lightfm_size}")
    
    # Step 4: Cross-validation (Placeholder)
    # ...
    
    # Step 5: Evaluate models
    print_separator("STEP 5: Model Evaluation")
    all_metrics = {'experiment_info': {'config': config}}
    
    # Evaluate PMF
    logger.info("\n--- Evaluating PMF Model ---")
    pmf_results = evaluator.evaluate_pmf(pmf_model, data['explicit']['testset'])
    all_metrics['pmf_results'] = {**pmf_results, 'model_info': pmf_model.model_info, 'model_size': pmf_size}
    
    # Evaluate BPR
    logger.info("\n--- Evaluating BPR-MF Model ---")
    bpr_results = evaluator.evaluate_bpr(bpr_model, data['implicit']['test_matrix'], data['implicit']['train_matrix'])
    all_metrics['bpr_results'] = {**bpr_results, 'model_info': bpr_model.model_info, 'model_size': bpr_size}

    # Evaluate LightFM
    if lightfm_model:
        logger.info("\n--- Evaluating LightFM Model ---")
        lightfm_results = evaluator.evaluate_bpr(lightfm_model, data['implicit']['test_matrix'], data['implicit']['train_matrix'])
        all_metrics['lightfm_results'] = {**lightfm_results, 'model_info': lightfm_model.model_info, 'model_size': lightfm_size}

    # Step 6: Generate example recommendations
    print_separator("STEP 6: Generating Example Recommendations")
    test_users = data['test_df']['user_id'].unique()[:3]
    for user_id in test_users:
        logger.info(f"\nRecommendations for User {user_id}:")
        
        # PMF recommendations
        pmf_recs = pmf_model.recommend(user_id, n_items=5)
        logger.info("PMF recommendations:")
        for item_id, score in pmf_recs:
            title = movies_df.loc[movies_df['item_id'] == item_id, 'title'].values[0]
            logger.info(f"  - {title} (Score: {score:.3f})")

        # BPR recommendations
        bpr_recs = bpr_model.recommend(user_id, n_items=5)
        logger.info("BPR recommendations:")
        for item_id, score in bpr_recs:
            title = movies_df.loc[movies_df['item_id'] == item_id, 'title'].values[0]
            logger.info(f"  - {title} (Score: {score:.3f})")

        # LightFM recommendations
        if lightfm_model:
            lightfm_recs = lightfm_model.recommend(user_id, n_items=5)
            logger.info("LightFM recommendations:")
            for item_id, score in lightfm_recs:
                title = movies_df.loc[movies_df['item_id'] == item_id, 'title'].values[0]
                logger.info(f"  - {title} (Score: {score:.3f})")

    # Step 7: Create visualizations
    print_separator("STEP 7: Creating Visualizations")
    # ... Visualization logic ...

    # Step 9: Save all results
    print_separator("STEP 9: Saving Results")
    
    all_metrics['experiment_info']['timestamp'] = datetime.now().isoformat()
    all_metrics['experiment_info']['total_execution_time'] = time.time() - start_time
    
    with open(results_dir / 'metrics' / 'comprehensive_results.json', 'w') as f:
        def convert(o):
            if isinstance(o, np.generic): return o.item()  
            if isinstance(o, (np.ndarray,)): return o.tolist()
            if isinstance(o, (Path,)): return str(o)
            raise TypeError
        json.dump(all_metrics, f, indent=2, default=convert)
    
    generate_summary_report(all_metrics, results_dir)
    
    # Final Summary
    elapsed_time = time.time() - start_time
    print_separator("EXECUTION COMPLETE")
    logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")
    logger.info(f"PMF RMSE: {pmf_results['rmse']:.4f}")
    logger.info(f"BPR Precision@10: {bpr_results.get('precision@10', 0):.4f}")
    if lightfm_model:
        logger.info(f"LightFM Precision@10: {all_metrics['lightfm_results'].get('precision@10', 0):.4f}")
    logger.info(f"Results saved to: {results_dir}")
    print_separator()

def generate_summary_report(metrics: Dict[str, Any], results_dir: Path):
    """Generate a markdown summary report."""
    report_path = results_dir / 'summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# MovieLens Recommender System - Results Summary\n\n")
        f.write(f"Generated: {metrics['experiment_info']['timestamp']}\n\n")
        
        f.write("## Model Performance\n\n")
        
        if 'pmf_results' in metrics:
            f.write("### PMF (Probabilistic Matrix Factorization)\n")
            f.write(f"- **RMSE**: {metrics['pmf_results'].get('rmse', 'N/A'):.4f}\n")
            f.write(f"- **MAE**: {metrics['pmf_results'].get('mae', 'N/A'):.4f}\n")
            f.write(f"- Training time: {metrics['pmf_results'].get('model_info', {}).get('training_time', 0):.2f}s\n\n")
        
        if 'bpr_results' in metrics:
            f.write("### BPR-MF (Bayesian Personalized Ranking)\n")
            f.write(f"- **Precision@10**: {metrics['bpr_results'].get('precision@10', 0):.4f}\n")
            f.write(f"- Recall@10: {metrics['bpr_results'].get('recall@10', 0):.4f}\n")
            f.write(f"- NDCG@10: {metrics['bpr_results'].get('ndcg@10', 0):.4f}\n\n")

        if 'lightfm_results' in metrics:
            f.write("### LightFM (Hybrid Model with Genre Features)\n")
            f.write(f"- **Precision@10**: {metrics['lightfm_results'].get('precision@10', 0):.4f}\n")
            f.write(f"- Recall@10: {metrics['lightfm_results'].get('recall@10', 0):.4f}\n")
            f.write(f"- NDCG@10: {metrics['lightfm_results'].get('ndcg@10', 0):.4f}\n\n")

        f.write("## Execution Details\n")
        f.write(f"- Total execution time: {metrics['experiment_info'].get('total_execution_time', 0)/60:.2f} minutes\n")
        
    logger.info(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()