# src/optimization/hyperparameter_tuning.py
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PercentilePruner
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Callable, Optional, List
import json
from pathlib import Path
import time
import joblib

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Advanced hyperparameter tuning using Optuna."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        self.best_params = None
        self.results_dir = Path(config['output']['results_dir']) / 'optimization'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def tune_pmf(self, train_data, val_data, 
                 n_trials: int = None, timeout: int = None,
                 n_jobs: int = 1) -> Dict[str, Any]:
        """Tune PMF hyperparameters with advanced Optuna features."""
        logger.info("Starting PMF hyperparameter tuning with Optuna...")
        
        n_trials = n_trials or self.config['optimization']['n_trials']
        timeout = timeout or self.config['optimization']['timeout']
        
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters with constraints
            params = {
                'n_factors': trial.suggest_int('n_factors', 10, 200, step=10),
                'n_epochs': trial.suggest_int('n_epochs', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'regularization': trial.suggest_float('regularization', 1e-4, 1e-1, log=True)
            }
            
            # Add conditional parameters
            use_biased = trial.suggest_categorical('use_biased', [True, False])
            params['biased'] = use_biased
            
            if use_biased:
                # Different regularization for biases
                params['reg_bu'] = trial.suggest_float('reg_bu', 1e-4, 1e-1, log=True)
                params['reg_bi'] = trial.suggest_float('reg_bi', 1e-4, 1e-1, log=True)
            
            # Log trial
            logger.info(f"Trial {trial.number}: {params}")
            
            # Train model with suggested parameters
            from src.models.pmf_model import PMFModel
            model_config = {'models': {'pmf': {**params, 'random_state': 42, 'verbose': False}}}
            
            try:
                model = PMFModel(model_config)
                model.fit(train_data)
                
                # Evaluate on validation set
                from src.evaluation.evaluator import ModelEvaluator
                evaluator = ModelEvaluator(self.config)
                results = evaluator.evaluate_pmf(model, val_data)
                
                # Store additional metrics
                trial.set_user_attr('mae', results['mae'])
                trial.set_user_attr('n_predictions', results['n_predictions'])
                
                # Return negative RMSE (Optuna maximizes by default)
                return -results['rmse']
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf')
        
        # Create study with advanced features
        sampler = TPESampler(seed=self.config['models']['pmf']['random_state'])
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            study_name='pmf_tuning',
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{self.results_dir}/pmf_study.db",
            load_if_exists=True
        )
        
        # Add default parameters as first trial
        default_params = {
            'n_factors': self.config['models']['pmf']['n_factors'],
            'n_epochs': self.config['models']['pmf']['n_epochs'],
            'learning_rate': self.config['models']['pmf']['learning_rate'],
            'regularization': self.config['models']['pmf']['regularization'],
            'use_biased': self.config['models']['pmf']['biased']
        }
        self.study.enqueue_trial(default_params)
        
        # Optimize
        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        logger.info(f"Best PMF parameters: {self.best_params}")
        logger.info(f"Best RMSE: {-self.study.best_value:.4f}")
        
        # Save results
        self._save_results('pmf')
        
        return self.best_params
    
    def tune_bpr(self, train_data, val_data,
                 n_trials: int = None, timeout: int = None,
                 n_jobs: int = 1) -> Dict[str, Any]:
        """Tune BPR hyperparameters with advanced features."""
        logger.info("Starting BPR-MF hyperparameter tuning with Optuna...")
        
        n_trials = n_trials or self.config['optimization']['n_trials']
        timeout = timeout or self.config['optimization']['timeout']
        
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = {
                'factors': trial.suggest_int('factors', 10, 200, step=10),
                'iterations': trial.suggest_int('iterations', 50, 300, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'regularization': trial.suggest_float('regularization', 1e-5, 1e-2, log=True)
            }
            
            # Algorithm variant
            algorithm = trial.suggest_categorical('algorithm', ['bpr', 'als', 'lmf'])
            params['algorithm'] = algorithm
            
            # Algorithm-specific parameters
            if algorithm == 'bpr':
                params['verify_negative_samples'] = trial.suggest_categorical(
                    'verify_negative_samples', [True, False]
                )
            
            # Add fixed parameters
            params.update({
                'num_threads': 0,
                'dtype': 'float32',
                'random_state': 42
            })
            
            logger.info(f"Trial {trial.number}: {params}")
            
            try:
                # Train model
                from src.models.bpr_model import BPRModel
                model_config = {'models': {'bpr': params}}
                model = BPRModel(model_config)
                model.fit(train_data)
                
                # Evaluate with early stopping
                from src.evaluation.evaluator import ModelEvaluator
                evaluator = ModelEvaluator(self.config)
                
                # Use smaller k for faster evaluation during tuning
                results = evaluator.evaluate_bpr(
                    model, val_data['test_matrix'], 
                    val_data['train_matrix'], k_values=[10],
                    calc_diversity=False  # Skip for speed
                )
                
                # Store additional metrics
                trial.set_user_attr('recall@10', results.get('recall@10', 0))
                trial.set_user_attr('ndcg@10', results.get('ndcg@10', 0))
                trial.set_user_attr('coverage', results.get('coverage', 0))
                
                # Prune if intermediate result is bad
                intermediate_value = results['precision@10']
                trial.report(intermediate_value, 0)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Return Precision@10
                return results['precision@10']
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                return 0.0
        
        # Create study
        sampler = TPESampler(seed=self.config['models']['bpr']['random_state'])
        pruner = PercentilePruner(25.0, n_startup_trials=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            study_name='bpr_tuning',
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{self.results_dir}/bpr_study.db",
            load_if_exists=True
        )
        
        # Optimize
        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        logger.info(f"Best BPR parameters: {self.best_params}")
        logger.info(f"Best Precision@10: {self.study.best_value:.4f}")
        
        # Save results
        self._save_results('bpr')
        
        return self.best_params
    
    def tune_ensemble(self, models: List[Any], train_data: Any, val_data: Any,
                     n_trials: int = 50) -> Dict[str, Any]:
        """Tune ensemble weights for combining multiple models."""
        logger.info("Tuning ensemble weights...")
        
        def objective(trial: Trial) -> float:
            # Suggest weights for each model
            weights = []
            for i in range(len(models)):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(models)] * len(models)
            
            # Make ensemble predictions
            ensemble_predictions = []
            
            for test_item in val_data:
                user_id, item_id, true_rating = test_item
                
                # Get predictions from each model
                model_preds = []
                for model in models:
                    pred = model.predict(user_id, item_id)
                    model_preds.append(pred)
                
                # Weighted average
                ensemble_pred = sum(w * p for w, p in zip(weights, model_preds))
                ensemble_predictions.append((true_rating, ensemble_pred))
            
            # Calculate RMSE
            from src.evaluation.metrics import RecommenderMetrics
            metrics = RecommenderMetrics()
            rmse = metrics.rmse(ensemble_predictions)
            
            return -rmse  # Minimize RMSE
        
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best weights
        best_weights = []
        for i in range(len(models)):
            weight = study.best_params[f'weight_{i}']
            best_weights.append(weight)
        
        # Normalize
        total = sum(best_weights)
        best_weights = [w / total for w in best_weights]
        
        logger.info(f"Best ensemble weights: {best_weights}")
        logger.info(f"Best ensemble RMSE: {-study.best_value:.4f}")
        
        return {'weights': best_weights, 'rmse': -study.best_value}
    
    def _save_results(self, model_name: str):
        """Save tuning results and visualizations."""
        # Save best parameters
        with open(self.results_dir / f'{model_name}_best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save study trials as dataframe
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(self.results_dir / f'{model_name}_trials.csv', index=False)
        
        # Save study object
        joblib.dump(self.study, self.results_dir / f'{model_name}_study.pkl')
        
        # Create optimization visualizations
        self._create_optimization_plots(model_name)
        
        # Save importance analysis
        importance = optuna.importance.get_param_importances(self.study)
        with open(self.results_dir / f'{model_name}_param_importance.json', 'w') as f:
            json.dump(importance, f, indent=2)
        
        logger.info(f"Optimization results saved to {self.results_dir}")
    
    def _create_optimization_plots(self, model_name: str):
        """Create optimization visualization plots."""
        import matplotlib.pyplot as plt
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_contour
        )
        
        # 1. Optimization history
        fig = plot_optimization_history(self.study)
        fig.write_image(self.results_dir / f"{model_name}_optimization_history.png")
        
        # 2. Parameter importance
        try:
            fig = plot_param_importances(self.study)
            fig.write_image(self.results_dir / f"{model_name}_param_importance.png")
        except:
            logger.warning("Could not create parameter importance plot")
        
        # 3. Parallel coordinate plot
        try:
            fig = plot_parallel_coordinate(self.study)
            fig.write_image(self.results_dir / f"{model_name}_parallel_coordinate.png")
        except:
            logger.warning("Could not create parallel coordinate plot")
        
        # 4. Contour plots for top 2 parameters
        try:
            importance = optuna.importance.get_param_importances(self.study)
            if len(importance) >= 2:
                top_params = list(importance.keys())[:2]
                fig = plot_contour(self.study, params=top_params)
                fig.write_image(self.results_dir / f"{model_name}_contour.png")
        except:
            logger.warning("Could not create contour plot")
        
        # 5. Custom convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        trials_df = self.study.trials_dataframe()
        best_values = trials_df['value'].cummax()
        
        ax.plot(trials_df.index, trials_df['value'], 'o', alpha=0.3, label='Trial values')
        ax.plot(trials_df.index, best_values, 'r-', linewidth=2, label='Best so far')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'{model_name.upper()} Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{model_name}_convergence.png", dpi=300)
        plt.close()
    
    def analyze_study(self, study_name: str) -> Dict[str, Any]:
        """Analyze a completed study."""
        study_path = self.results_dir / f"{study_name}_study.pkl"
        
        if not study_path.exists():
            logger.error(f"Study file not found: {study_path}")
            return {}
        
        study = joblib.load(study_path)
        
        analysis = {
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'param_importance': optuna.importance.get_param_importances(study)
        }
        
        # Get parameter ranges actually explored
        trials_df = study.trials_dataframe()
        param_ranges = {}
        for param in study.best_params.keys():
            if f'params_{param}' in trials_df.columns:
                param_ranges[param] = {
                    'min': trials_df[f'params_{param}'].min(),
                    'max': trials_df[f'params_{param}'].max(),
                    'mean': trials_df[f'params_{param}'].mean(),
                    'std': trials_df[f'params_{param}'].std()
                }
        
        analysis['param_ranges'] = param_ranges
        
        return analysis