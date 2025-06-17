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
from src.models.pmf_model import PMFModel
from src.evaluation.evaluator import ModelEvaluator
from src.models.bpr_model import BPRModel
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        self.best_params = None
        self.results_dir = Path(config['output']['results_dir']) / 'optimization'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def tune_pmf(self, train_data, val_data, 
                 n_trials: int = None, timeout: int = None,
                 n_jobs: int = 1) -> Dict[str, Any]:
        
        logger.info(" hyperparameter tuning with Optuna")
        
        n_trials = n_trials or self.config['optimization']['n_trials']
        timeout = timeout or self.config['optimization']['timeout']
        
        def objective(trial: Trial) -> float:
            params = {
                'n_factors': trial.suggest_int('n_factors', 10, 200, step=10),
                'n_epochs': trial.suggest_int('n_epochs', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'regularization': trial.suggest_float('regularization', 1e-4, 1e-1, log=True)
            }
            
            use_biased = trial.suggest_categorical('use_biased', [True, False])
            params['biased'] = use_biased
            
            if use_biased:
                params['reg_bu'] = trial.suggest_float('reg_bu', 1e-4, 1e-1, log=True)
                params['reg_bi'] = trial.suggest_float('reg_bi', 1e-4, 1e-1, log=True)
            
            
            model_config = {'models': {'pmf': {**params, 'random_state': 42, 'verbose': False}}}
            
            model = PMFModel(model_config)
            model.fit(train_data)
                
            evaluator = ModelEvaluator(self.config)
            results = evaluator.evaluate_pmf(model, val_data)
                
            trial.set_user_attr('mae', results['mae'])
            trial.set_user_attr('n_predictions', results['n_predictions'])
                
            return -results['rmse']
                
        
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
        
        default_params = {
            'n_factors': self.config['models']['pmf']['n_factors'],
            'n_epochs': self.config['models']['pmf']['n_epochs'],
            'learning_rate': self.config['models']['pmf']['learning_rate'],
            'regularization': self.config['models']['pmf']['regularization'],
            'use_biased': self.config['models']['pmf']['biased']
        }
        self.study.enqueue_trial(default_params)
        
        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        logger.info(f"Best PMF parameters: {self.best_params}")
        logger.info(f"Best RMSE: {-self.study.best_value:.4f}")
        self._save_results('pmf')
        
        return self.best_params
    
    def tune_bpr(self, train_data, val_data,
                 n_trials: int = None, timeout: int = None,
                 n_jobs: int = 1) -> Dict[str, Any]:
        logger.info("BPR-MF hyperparameter tuning")
        
        n_trials = n_trials or self.config['optimization']['n_trials']
        timeout = timeout or self.config['optimization']['timeout']
        
        def objective(trial: Trial) -> float:
            params = {
                'factors': trial.suggest_int('factors', 10, 200, step=10),
                'iterations': trial.suggest_int('iterations', 50, 300, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'regularization': trial.suggest_float('regularization', 1e-5, 1e-2, log=True)
            }
            
            algorithm = trial.suggest_categorical('algorithm', ['bpr', 'als', 'lmf'])
            params['algorithm'] = algorithm
            
            if algorithm == 'bpr':
                params['verify_negative_samples'] = trial.suggest_categorical(
                    'verify_negative_samples', [True, False]
                )
            
            params.update({
                'num_threads': 0,
                'dtype': 'float32',
                'random_state': 42
            })
            
            
            model_config = {'models': {'bpr': params}}
            model = BPRModel(model_config)
            model.fit(train_data)
                
            evaluator = ModelEvaluator(self.config)
                
            results = evaluator.evaluate_bpr(
                model, val_data['test_matrix'], 
                val_data['train_matrix'], k_values=[10],
                calc_diversity=False  # Skip for speed
            )
                
            trial.set_user_attr('recall@10', results.get('recall@10', 0))
            trial.set_user_attr('ndcg@10', results.get('ndcg@10', 0))
            trial.set_user_attr('coverage', results.get('coverage', 0))
                
            intermediate_value = results['precision@10']
            trial.report(intermediate_value, 0)
                
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            return results['precision@10']
        
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
        
        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        logger.info(f"Best BPR parameters: {self.best_params}")
        logger.info(f"Best Precision@10: {self.study.best_value:.4f}")
        
        self._save_results('bpr')
        
        return self.best_params
    
    def _save_results(self, model_name: str):
        with open(self.results_dir / f'{model_name}_best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(self.results_dir / f'{model_name}_trials.csv', index=False)
        
        joblib.dump(self.study, self.results_dir / f'{model_name}_study.pkl')
        
        
        importance = optuna.importance.get_param_importances(self.study)
        with open(self.results_dir / f'{model_name}_param_importance.json', 'w') as f:
            json.dump(importance, f, indent=2)
        
        logger.info(f"Optimization results saved to {self.results_dir}")