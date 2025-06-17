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
from src.models.pmf_model import PMFModel
from src.models.bpr_model import BPRModel
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
    logger.info("Advanced Features: Hyperparameter Tuning, Cross-validation, Ensemble")
    print_separator()
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory structure
    results_dir = Path(config['output']['results_dir'])
    for subdir in ['models', 'metrics', 'plots', 'predictions', 'optimization']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_loader = MovieLensDataLoader('config/config.yaml')
    evaluator = ModelEvaluator(config)
    visualizer = RecommenderVisualizer(str(results_dir / 'plots'))
    
    # Step 1: Load and prepare data
    print_separator("STEP 1: Data Loading and Preparation")
    
    data = data_loader.prepare_data_splits()
    movies_df = data_loader.load_movies_with_genres()
    users_df = data_loader.load_user_info()
    
    # Print comprehensive dataset statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total ratings: {len(data['ratings_df']):,}")
    logger.info(f"Train ratings: {len(data['train_df']):,}")
    logger.info(f"Test ratings: {len(data['test_df']):,}")
    logger.info(f"Number of users: {data['ratings_df']['user_id'].nunique():,}")
    logger.info(f"Number of items: {data['ratings_df']['item_id'].nunique():,}")
    logger.info(f"Rating density: {len(data['ratings_df']) / (data['ratings_df']['user_id'].nunique() * data['ratings_df']['item_id'].nunique()):.4%}")
    logger.info(f"Average ratings per user: {len(data['ratings_df']) / data['ratings_df']['user_id'].nunique():.2f}")
    logger.info(f"Average ratings per item: {len(data['ratings_df']) / data['ratings_df']['item_id'].nunique():.2f}")
    
    # Genre statistics
    genre_counts = movies_df['primary_genre'].value_counts()
    logger.info(f"\nTop 5 genres: {genre_counts.head().to_dict()}")
    
    # User demographics
    logger.info(f"\nUser demographics:")
    logger.info(f"Age range: {users_df['age'].min()}-{users_df['age'].max()}")
    logger.info(f"Gender distribution: {users_df['gender'].value_counts().to_dict()}")
    
    # Step 2: Hyperparameter Tuning (if enabled)
    if config['optimization']['run_optimization']:
        print_separator("STEP 2: Hyperparameter Optimization")
        
        tuner = HyperparameterTuner(config)
        
        # Create validation splits
        train_split, val_split = data_loader.create_validation_split(
            data['train_df'], val_ratio=0.1
        )
        
        # Prepare validation data
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1, 5))
        
        # PMF validation data
        pmf_train_data = Dataset.load_from_df(
            train_split[['user_id', 'item_id', 'rating']], reader
        ).build_full_trainset()
        pmf_val_data = [(row.user_id, row.item_id, row.rating) 
                        for row in val_split.itertuples(index=False)]
        
        # Tune PMF
        logger.info("\n--- Tuning PMF Hyperparameters ---")
        best_pmf_params = tuner.tune_pmf(
            pmf_train_data, 
            pmf_val_data[:1000],  # Use subset for efficiency
            n_trials=config['optimization'].get('n_trials', 50),
            n_jobs=config['optimization'].get('n_jobs', 1)
        )
        
        # Update config with best parameters
        config['models']['pmf'].update(best_pmf_params)
        
        # BPR validation data
        # Would need to prepare implicit validation data here
                # ----------------------- BPR validation data -----------------------
        # Build an implicit (binary) train / val set that mirrors the explicit
        # split we just made. We reuse the helper already defined in
        # MovieLensDataLoader.
        bpr_val_data = data_loader._prepare_implicit_data(
            data['ratings_df'],          # full DF (needed for id mapping)
            train_split,                 # implicit-train matrix
            val_split,                   # implicit-validation “test” matrix
            threshold=config['data']['min_rating_threshold']
        )

        # --- Tuning BPR Hyperparameters ---
        logger.info("\n--- Tuning BPR Hyperparameters ---")
        best_bpr_params = tuner.tune_bpr(
            bpr_val_data,               # train_data (implicit)
            bpr_val_data,               # val_data   (implicit)
            n_trials=config['optimization'].get('n_trials', 50),
            n_jobs  =config['optimization'].get('n_jobs',   1)
        )

        # Update the live config so the **final** BPR model is trained
        # with the best params we just found.
        config['models']['bpr'].update(best_bpr_params)

        
    else:
        print_separator("STEP 2: Training Models with Default Parameters")
    
    # Step 3: Train models
    print_separator("STEP 3: Training Models")
    
    # Train PMF
    logger.info("\n--- Training PMF Model ---")
    pmf_config = config['models']['pmf']
    logger.info(f"PMF Configuration: {pmf_config}")
    
    pmf_model = PMFModel(config)
    pmf_model.fit(data['explicit']['trainset'])
    
    # Get model size info
    pmf_size = pmf_model.get_model_size()
    logger.info(f"PMF model size: {pmf_size}")
    
    # Train BPR
    logger.info("\n--- Training BPR-MF Model ---")
    bpr_config = config['models']['bpr']
    logger.info(f"BPR Configuration: {bpr_config}")
    
    bpr_model = BPRModel(config)
    bpr_model.fit(data['implicit'])
    
    bpr_size = bpr_model.get_model_size()
    logger.info(f"BPR model size: {bpr_size}")
    
    # Step 4: Cross-validation (if enabled)
    if config['advanced']['cross_validation']['enabled']:
        print_separator("STEP 4: Cross-Validation")
        
        n_folds = config['advanced']['cross_validation']['n_folds']
        
        # Cross-validate PMF
        logger.info(f"\n--- {n_folds}-Fold Cross-Validation for PMF ---")
        pmf_cv_results = evaluator.cross_validate_pmf(
            PMFModel, data, n_folds=n_folds
        )
        logger.info(f"PMF CV Results: RMSE = {pmf_cv_results['rmse_mean']:.4f} "
                   f"(±{pmf_cv_results['rmse_std']:.4f})")
    
    # Step 5: Evaluate models
    print_separator("STEP 5: Model Evaluation")
    
    # Evaluate PMF
    logger.info("\n--- Evaluating PMF Model ---")
    pmf_results = evaluator.evaluate_pmf(
        pmf_model, 
        data['explicit']['testset'],
        calc_confidence=config['evaluation']['calc_confidence']
    )
    
    # Evaluate BPR
    logger.info("\n--- Evaluating BPR-MF Model ---")
    bpr_results = evaluator.evaluate_bpr(
        bpr_model, 
        data['implicit']['test_matrix'],
        data['implicit']['train_matrix'],
        k_values=config['evaluation']['k_values'],
        calc_diversity=config['evaluation']['calc_diversity']
    )
    
    # Time-based evaluation (if enabled)
    if config['advanced']['time_based_evaluation']['enabled']:
        logger.info("\n--- Time-based Evaluation ---")
        time_results = evaluator.evaluate_time_based_split(
            pmf_model, 
            data['test_df'],
            time_windows=config['advanced']['time_based_evaluation']['n_windows']
        )
        logger.info(f"RMSE over time: {time_results['window_rmse']}")
    
    # Step 6: Generate recommendations
    print_separator("STEP 6: Generating Example Recommendations")
    
    # Sample some test users
    test_users = data['test_df']['user_id'].unique()[:5]
    
    all_recommendations = {}
    for user_id in test_users:
        logger.info(f"\nRecommendations for User {user_id}:")
        
        # PMF recommendations
        pmf_recs = pmf_model.recommend(user_id, n_items=10)
        logger.info("PMF recommendations:")
        for i, (item_id, score) in enumerate(pmf_recs[:5]):
            movie = movies_df[movies_df['item_id'] == item_id]
            if not movie.empty:
                title = movie.iloc[0]['title']
                genre = movie.iloc[0]['primary_genre']
                logger.info(f"  {i+1}. {title} ({genre}) - Score: {score:.3f}")
        
        # BPR recommendations
        bpr_recs = bpr_model.recommend(user_id, n_items=10)
        logger.info("BPR recommendations:")
        for i, (item_id, score) in enumerate(bpr_recs[:5]):
            movie = movies_df[movies_df['item_id'] == item_id]
            if not movie.empty:
                title = movie.iloc[0]['title']
                genre = movie.iloc[0]['primary_genre']
                logger.info(f"  {i+1}. {title} ({genre}) - Score: {score:.3f}")
        
        all_recommendations[user_id] = bpr_recs
    
    # Find similar items example
    logger.info("\n--- Finding Similar Items ---")
    popular_movie_id = data['ratings_df']['item_id'].value_counts().index[0]
    popular_movie = movies_df[movies_df['item_id'] == popular_movie_id].iloc[0]
    logger.info(f"Finding items similar to: {popular_movie['title']}")
    
    similar_items = pmf_model.get_similar_items(popular_movie_id, n_similar=5)
    for item_id, similarity in similar_items:
        movie = movies_df[movies_df['item_id'] == item_id]
        if not movie.empty:
            logger.info(f"  - {movie.iloc[0]['title']} (similarity: {similarity:.3f})")
    
    # # Step 7: Create visualizations
    # print_separator("STEP 7: Creating Visualizations")
    
    # # Get factors
    # pmf_user_factors, pmf_item_factors = pmf_model.get_user_factors(), pmf_model.get_item_factors()
    # bpr_user_factors, bpr_item_factors = bpr_model.get_user_factors(), bpr_model.get_item_factors()
    
    # # Map genres to items
    # item_genres = []
    # for item_idx in range(len(pmf_item_factors)):
    #     item_id = data['explicit']['trainset'].to_raw_iid(item_idx)
    #     movie = movies_df[movies_df['item_id'] == item_id]
    #     if not movie.empty:
    #         item_genres.append(movie.iloc[0]['primary_genre'])
    #     else:
    #         item_genres.append('unknown')
    
    # # Create comprehensive visualizations
    # logger.info("Creating factor analysis plots...")
    # visualizer.plot_factor_analysis(
    #     pmf_item_factors, bpr_item_factors, item_genres
    # )
    
    # logger.info("Creating performance comparison plots...")
    # visualizer.plot_performance_comparison(pmf_results, bpr_results)
    
    # logger.info("Creating recommendation distribution analysis...")
    # visualizer.plot_recommendation_distribution(
    #     all_recommendations, movies_df
    # )
    
    # # Plot hyperparameter tuning results if available
    # if config['optimization']['run_optimization']:
    #     logger.info("Creating hyperparameter analysis plots...")
    #     pmf_trials_df = pd.read_csv(results_dir / 'optimization' / 'pmf_trials.csv')
    #     visualizer.plot_hyperparameter_analysis(pmf_trials_df, save_name="pmf_hyperparameter_analysis")
    
    # Step 8: Ensemble methods (if enabled)
    if config['advanced']['ensemble']['enabled']:
        print_separator("STEP 8: Ensemble Methods")
        
        logger.info("Training ensemble model...")
        
        if config['advanced']['ensemble']['method'] == 'weighted':
            # Simple weighted ensemble
            ensemble_weights = {'pmf': 0.5, 'bpr': 0.5}
            
            # Optionally tune weights
            if config['optimization']['run_optimization']:
                tuner = HyperparameterTuner(config)
                ensemble_result = tuner.tune_ensemble(
                    [pmf_model, bpr_model],
                    data['explicit']['testset'][:1000],
                    n_trials=50
                )
                ensemble_weights = {
                    'pmf': ensemble_result['weights'][0],
                    'bpr': ensemble_result['weights'][1]
                }
            
            logger.info(f"Ensemble weights: {ensemble_weights}")
    
    # Step 9: Save all results
    print_separator("STEP 9: Saving Results")
    
    # Save models
    if config['output']['save_models']:
        pmf_model.save_model(results_dir / 'models' / 'pmf_model.pkl')
        bpr_model.save_model(results_dir / 'models' / 'bpr_model.pkl')
    
    # Prepare comprehensive metrics
    all_metrics = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': time.time() - start_time,
            'config': config
        },
        'dataset_info': {
            'name': 'MovieLens-100K',
            'total_ratings': len(data['ratings_df']),
            'train_size': len(data['train_df']),
            'test_size': len(data['test_df']),
            'n_users': data['ratings_df']['user_id'].nunique(),
            'n_items': data['ratings_df']['item_id'].nunique(),
            'density': len(data['ratings_df']) / (data['ratings_df']['user_id'].nunique() * 
                                                 data['ratings_df']['item_id'].nunique()),
            'temporal_split': config['data']['temporal_split'],
            'implicit_threshold': config['data']['min_rating_threshold']
        },
        'pmf_results': {
            **pmf_results,
            'model_info': pmf_model.model_info,
            'model_size': pmf_size,
            'algorithm': config['models']['pmf']['algorithm']
        },
        'bpr_results': {
            **bpr_results,
            'model_info': bpr_model.model_info,
            'model_size': bpr_size,
            'algorithm': config['models']['bpr']['algorithm']
        }
    }
    
    # Add cross-validation results if available
    if config['advanced']['cross_validation']['enabled']:
        all_metrics['cross_validation'] = {
            'pmf': pmf_cv_results
        }
    
    # Add optimization results if available
    if config['optimization']['run_optimization']:
        all_metrics['optimization'] = {
            'pmf_best_params': best_pmf_params,
            'pmf_best_rmse': -tuner.study.best_value
        }
    
    # Save comprehensive metrics
    with open(results_dir / 'metrics' / 'comprehensive_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_metrics, results_dir)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_separator("EXECUTION COMPLETE")
    logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")
    logger.info(f"PMF RMSE: {pmf_results['rmse']:.4f}")
    logger.info(f"BPR Precision@10: {bpr_results.get('precision@10', 0):.4f}")
    logger.info(f"Results saved to: {results_dir}")
    print_separator()

def generate_summary_report(metrics: Dict[str, Any], results_dir: Path):
    """Generate a markdown summary report."""
    report_path = results_dir / 'summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# MovieLens Recommender System - Results Summary\n\n")
        f.write(f"Generated: {metrics['experiment_info']['timestamp']}\n\n")
        
        f.write("## Dataset Information\n")
        f.write(f"- Total ratings: {metrics['dataset_info']['total_ratings']:,}\n")
        f.write(f"- Train/Test split: {metrics['dataset_info']['train_size']:,} / "
                f"{metrics['dataset_info']['test_size']:,}\n")
        f.write(f"- Users: {metrics['dataset_info']['n_users']:,}\n")
        f.write(f"- Items: {metrics['dataset_info']['n_items']:,}\n")
        f.write(f"- Density: {metrics['dataset_info']['density']:.4%}\n\n")
        
        f.write("## Model Performance\n\n")
        
        f.write("### PMF (Probabilistic Matrix Factorization)\n")
        f.write(f"- RMSE: {metrics['pmf_results']['rmse']:.4f}\n")
        f.write(f"- MAE: {metrics['pmf_results']['mae']:.4f}\n")
        f.write(f"- Training time: {metrics['pmf_results']['model_info']['training_time']:.2f}s\n")
        f.write(f"- Factors: {metrics['pmf_results']['model_size']['n_factors']}\n\n")
        
        f.write("### BPR-MF (Bayesian Personalized Ranking)\n")
        f.write(f"- Precision@10: {metrics['bpr_results'].get('precision@10', 0):.4f}\n")
        f.write(f"- Recall@10: {metrics['bpr_results'].get('recall@10', 0):.4f}\n")
        f.write(f"- NDCG@10: {metrics['bpr_results'].get('ndcg@10', 0):.4f}\n")
        f.write(f"- Coverage: {metrics['bpr_results'].get('coverage', 0):.4f}\n")
        f.write(f"- Training time: {metrics['bpr_results']['model_info']['training_time']:.2f}s\n\n")
        
        f.write("## Execution Details\n")
        f.write(f"- Total execution time: {metrics['experiment_info']['total_execution_time']/60:.2f} minutes\n")
        
    logger.info(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()