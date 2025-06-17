# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

class RecommenderVisualizer:
    """Advanced visualization tools for recommender systems."""
    
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def plot_factor_analysis(self, pmf_factors: np.ndarray, 
                           bpr_factors: np.ndarray,
                           genres: List[str],
                           save_name: str = "factor_analysis"):
        """Create comprehensive factor analysis visualization."""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Ensure same dimensionality
        min_factors = min(pmf_factors.shape[1], bpr_factors.shape[1])
        pmf_factors = pmf_factors[:, :min_factors]
        bpr_factors = bpr_factors[:, :min_factors]
        
        # 1. PCA visualization for PMF
        ax1 = fig.add_subplot(gs[0, 0])
        pca = PCA(n_components=2)
        pmf_2d = pca.fit_transform(pmf_factors)
        
        # Create genre color mapping
        unique_genres = list(set(genres))[:10]  # Limit to 10 genres for clarity
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = {g: colors[i] for i, g in enumerate(unique_genres)}
        
        # Plot PMF
        for genre in unique_genres:
            mask = [g == genre for g in genres]
            if any(mask):
                ax1.scatter(pmf_2d[mask, 0], pmf_2d[mask, 1], 
                           c=[genre_to_color[genre]], label=genre, alpha=0.6, s=30)
        
        ax1.set_title('PMF Item Factors (PCA)', fontsize=14)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Keep only items present in BOTH models
        common = min(pmf_factors.shape[0], bpr_factors.shape[0])
        pmf_factors, bpr_factors, genres = pmf_factors[:common], bpr_factors[:common], genres[:common]

        # 2. PCA visualization for BPR
        ax2 = fig.add_subplot(gs[0, 1])
        bpr_2d = pca.fit_transform(bpr_factors)
        
        for genre in unique_genres:
            mask = [g == genre for g in genres]
            if any(mask):
                ax2.scatter(bpr_2d[mask, 0], bpr_2d[mask, 1], 
                           c=[genre_to_color[genre]], label=genre, alpha=0.6, s=30)
        
        ax2.set_title('BPR-MF Item Factors (PCA)', fontsize=14)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # 3. t-SNE visualization for PMF
        ax3 = fig.add_subplot(gs[0, 2])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # Sample items if too many
        n_samples = min(1000, len(pmf_factors))
        sample_indices = np.random.choice(len(pmf_factors), n_samples, replace=False)
        pmf_sample = pmf_factors[sample_indices]
        genres_sample = [genres[i] for i in sample_indices]
        
        pmf_tsne = tsne.fit_transform(pmf_sample)
        
        for genre in unique_genres:
            mask = [g == genre for g in genres_sample]
            if any(mask):
                ax3.scatter(pmf_tsne[mask, 0], pmf_tsne[mask, 1], 
                           c=[genre_to_color[genre]], label=genre, alpha=0.6, s=30)
        
        ax3.set_title('PMF Item Factors (t-SNE)', fontsize=14)
        ax3.set_xlabel('t-SNE 1')
        ax3.set_ylabel('t-SNE 2')
        
        # 4. Factor correlation heatmap for PMF
        ax4 = fig.add_subplot(gs[1, 0])
        pmf_corr = np.corrcoef(pmf_factors[:, :15].T)
        sns.heatmap(pmf_corr, annot=False, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax4, cbar_kws={'label': 'Correlation'})
        ax4.set_title('PMF Factor Correlations (Top 15)', fontsize=14)
        ax4.set_xlabel('Factor')
        ax4.set_ylabel('Factor')
        
        # 5. Factor correlation heatmap for BPR
        ax5 = fig.add_subplot(gs[1, 1])
        bpr_corr = np.corrcoef(bpr_factors[:, :15].T)
        sns.heatmap(bpr_corr, annot=False, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax5, cbar_kws={'label': 'Correlation'})
        ax5.set_title('BPR-MF Factor Correlations (Top 15)', fontsize=14)
        ax5.set_xlabel('Factor')
        ax5.set_ylabel('Factor')
        
        # 6. Factor importance (variance explained)
        ax6 = fig.add_subplot(gs[1, 2])
        pmf_var = np.var(pmf_factors, axis=0)
        bpr_var = np.var(bpr_factors, axis=0)
        
        x = np.arange(min(20, len(pmf_var)))
        width = 0.35
        
        ax6.bar(x - width/2, pmf_var[:20], width, label='PMF', alpha=0.7)
        ax6.bar(x + width/2, bpr_var[:20], width, label='BPR-MF', alpha=0.7)
        
        ax6.set_title('Factor Importance (Variance)', fontsize=14)
        ax6.set_xlabel('Factor Index')
        ax6.set_ylabel('Variance')
        ax6.legend()
        
        # 7. Genre distribution in factor space
        ax7 = fig.add_subplot(gs[2, :2])
        genre_counts = pd.Series(genres).value_counts().head(15)
        ax7.barh(genre_counts.index, genre_counts.values)
        ax7.set_title('Genre Distribution in Dataset', fontsize=14)
        ax7.set_xlabel('Number of Items')
        
        # 8. Factor norms distribution
        ax8 = fig.add_subplot(gs[2, 2])
        pmf_norms = np.linalg.norm(pmf_factors, axis=1)
        bpr_norms = np.linalg.norm(bpr_factors, axis=1)
        
        ax8.hist(pmf_norms, bins=50, alpha=0.5, label='PMF', density=True)
        ax8.hist(bpr_norms, bins=50, alpha=0.5, label='BPR-MF', density=True)
        ax8.set_title('Distribution of Factor Norms', fontsize=14)
        ax8.set_xlabel('L2 Norm')
        ax8.set_ylabel('Density')
        ax8.legend()
        
        plt.suptitle('Comprehensive Factor Analysis', fontsize=16)
        
        # Save in multiple formats
        for fmt in ['png', 'pdf']:
            filepath = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(filepath, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight')
        
        plt.close()
        logger.info(f"Factor analysis plots saved to {self.output_dir}")
    
    def plot_performance_comparison(self, pmf_results: Dict[str, float],
                                  bpr_results: Dict[str, float],
                                  save_name: str = "performance_comparison"):
        """Create performance comparison visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. RMSE/MAE comparison for PMF
        ax1 = axes[0, 0]
        pmf_metrics = ['RMSE', 'MAE']
        pmf_values = [pmf_results['rmse'], pmf_results['mae']]
        
        bars1 = ax1.bar(pmf_metrics, pmf_values, color=['#1f77b4', '#ff7f0e'])
        ax1.set_title('PMF Performance Metrics', fontsize=14)
        ax1.set_ylabel('Error')
        ax1.set_ylim(0, max(pmf_values) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars1, pmf_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 2. Precision/Recall at different K values for BPR
        ax2 = axes[0, 1]
        k_values = [5, 10, 20]
        precisions = [bpr_results.get(f'precision@{k}', 0) for k in k_values]
        recalls = [bpr_results.get(f'recall@{k}', 0) for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, precisions, width, label='Precision', 
                       color='#2ca02c')
        bars3 = ax2.bar(x + width/2, recalls, width, label='Recall', 
                       color='#d62728')
        
        ax2.set_title('BPR-MF Precision/Recall @K', fontsize=14)
        ax2.set_xlabel('K')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(k_values)
        ax2.legend()
        ax2.set_ylim(0, max(max(precisions), max(recalls)) * 1.2)
        
        # 3. NDCG comparison
        ax3 = axes[0, 2]
        ndcg_values = [bpr_results.get(f'ndcg@{k}', 0) for k in k_values]
        
        bars4 = ax3.bar([f'NDCG@{k}' for k in k_values], ndcg_values, 
                       color='#9467bd')
        ax3.set_title('BPR-MF NDCG Performance', fontsize=14)
        ax3.set_ylabel('NDCG Score')
        ax3.set_ylim(0, max(ndcg_values) * 1.2)
        
        # Add value labels
        for bar, value in zip(bars4, ndcg_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4. F1 scores
        ax4 = axes[1, 0]
        f1_values = [bpr_results.get(f'f1@{k}', 0) for k in k_values]
        
        ax4.plot(k_values, f1_values, 'o-', markersize=8, linewidth=2)
        ax4.set_title('BPR-MF F1 Score @K', fontsize=14)
        ax4.set_xlabel('K')
        ax4.set_ylabel('F1 Score')
        ax4.grid(True, alpha=0.3)
        
        # 5. Coverage and Diversity
        ax5 = axes[1, 1]
        other_metrics = ['Coverage', 'Diversity', 'Novelty']
        other_values = [
            bpr_results.get('coverage', 0),
            bpr_results.get('diversity', 0),
            bpr_results.get('novelty', 0)
        ]
        
        bars5 = ax5.bar(other_metrics, other_values, color=['#e377c2', '#7f7f7f', '#bcbd22'])
        ax5.set_title('Additional BPR-MF Metrics', fontsize=14)
        ax5.set_ylabel('Score')
        ax5.set_ylim(0, max(other_values) * 1.2)
        
        # Add value labels
        for bar, value in zip(bars5, other_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Evaluation time comparison
        ax6 = axes[1, 2]
        eval_times = [
            pmf_results.get('evaluation_time', 0),
            bpr_results.get('evaluation_time', 0)
        ]
        
        bars6 = ax6.bar(['PMF', 'BPR-MF'], eval_times, color=['#1f77b4', '#ff7f0e'])
        ax6.set_title('Evaluation Time Comparison', fontsize=14)
        ax6.set_ylabel('Time (seconds)')
        
        for bar, value in zip(bars6, eval_times):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}s', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(filepath, dpi=300 if fmt == 'png' else None)
        
        plt.close()
        logger.info(f"Performance comparison plots saved to {self.output_dir}")
    
    def plot_hyperparameter_analysis(self, tuning_results: pd.DataFrame,
                                save_name: str = "hyperparameter_analysis"):
        """Visualize hyperparameter tuning results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Filter for numeric columns only
        param_cols = [col for col in tuning_results.columns 
                    if col not in ['value', 'datetime_start', 'datetime_complete']]
        
        # Identify numeric parameter columns
        numeric_param_cols = []
        for col in param_cols:
            try:
                pd.to_numeric(tuning_results[col], errors='raise')
                numeric_param_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        # 1. Factors vs Performance heatmap
        ax1 = axes[0, 0]
        if 'factors' in numeric_param_cols and 'learning_rate' in numeric_param_cols:
            try:
                pivot1 = tuning_results.pivot_table(
                    values='value', index='factors', columns='learning_rate'
                )
                sns.heatmap(pivot1, annot=True, fmt='.4f', cmap='viridis', ax=ax1)
                ax1.set_title('Performance by Factors and Learning Rate', fontsize=14)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Heatmap unavailable\n({str(e)[:30]})', 
                        transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Performance Heatmap', fontsize=14)
        else:
            ax1.text(0.5, 0.5, 'Insufficient numeric\nparameters for heatmap', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Performance Heatmap', fontsize=14)
        
        # 2. Learning curve
        ax2 = axes[0, 1]
        if 'value' in tuning_results.columns:
            ax2.plot(tuning_results.index, tuning_results['value'], 'b-', alpha=0.3)
            
            # Highlight best trial
            best_idx = tuning_results['value'].idxmax()
            ax2.scatter(best_idx, tuning_results.loc[best_idx, 'value'], 
                    color='red', s=100, zorder=5, label='Best')
            
            # Add moving average
            window = min(10, len(tuning_results) // 5)
            if window > 1:
                ma = tuning_results['value'].rolling(window=window).mean()
                ax2.plot(tuning_results.index, ma, 'g-', linewidth=2, label=f'MA({window})')
            
            ax2.set_title('Optimization Progress', fontsize=14)
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Objective Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No optimization values available', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Optimization Progress', fontsize=14)
        
        # 3. Parameter distributions (numeric only)
        ax3 = axes[0, 2]
        if numeric_param_cols:
            try:
                tuning_results[numeric_param_cols].boxplot(ax=ax3)
                ax3.set_title('Parameter Distributions', fontsize=14)
                ax3.set_ylabel('Parameter Value')
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            except Exception:
                ax3.text(0.5, 0.5, 'Parameter distributions\nunavailable', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Parameter Distributions', fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'No numeric parameters\nfor distribution plot', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Parameter Distributions', fontsize=14)
        
        # 4. Parallel coordinates plot (numeric parameters only)
        ax4 = axes[1, 0]
        if len(numeric_param_cols) >= 2:
            try:
                # Normalize numeric parameters only
                normalized_df = tuning_results[numeric_param_cols].copy()
                for col in numeric_param_cols:
                    col_data = pd.to_numeric(tuning_results[col], errors='coerce')
                    col_min = col_data.min()
                    col_max = col_data.max()
                    if col_max > col_min and not (pd.isna(col_min) or pd.isna(col_max)):
                        normalized_df[col] = (col_data - col_min) / (col_max - col_min)
                    else:
                        normalized_df[col] = 0.5
                
                # Color by performance
                if 'value' in tuning_results.columns:
                    values = pd.to_numeric(tuning_results['value'], errors='coerce')
                    valid_values = ~values.isna()
                    if valid_values.sum() > 0:
                        colors = plt.cm.viridis(values[valid_values] / values[valid_values].max())
                        
                        for idx, (_, row) in enumerate(normalized_df[valid_values].iterrows()):
                            ax4.plot(range(len(numeric_param_cols)), row.values, 
                                    color=colors[idx], alpha=0.5, linewidth=1)
                
                ax4.set_xticks(range(len(numeric_param_cols)))
                ax4.set_xticklabels(numeric_param_cols, rotation=45)
                ax4.set_title('Parallel Coordinates - Hyperparameters', fontsize=14)
                ax4.set_ylabel('Normalized Value')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Parallel coordinates\nunavailable', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Parallel Coordinates - Hyperparameters', fontsize=14)
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 numeric\nparameters for plot', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Parallel Coordinates - Hyperparameters', fontsize=14)
        
        # 5. Best parameters visualization
        ax5 = axes[1, 1]
        if not tuning_results.empty and 'value' in tuning_results.columns:
            try:
                best_idx = tuning_results['value'].idxmax()
                if numeric_param_cols:
                    best_params = tuning_results.loc[best_idx, numeric_param_cols]
                    ax5.barh(best_params.index, best_params.values)
                    ax5.set_title('Best Hyperparameters', fontsize=14)
                    ax5.set_xlabel('Parameter Value')
                    
                    # Add value labels
                    for i, (param, value) in enumerate(best_params.items()):
                        ax5.text(value, i, f' {value:.4f}', va='center')
                else:
                    ax5.text(0.5, 0.5, 'No numeric parameters\nfor best params plot', 
                            transform=ax5.transAxes, ha='center', va='center')
                    ax5.set_title('Best Hyperparameters', fontsize=14)
            except Exception:
                ax5.text(0.5, 0.5, 'Best parameters\nplot unavailable', 
                        transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title('Best Hyperparameters', fontsize=14)
        else:
            ax5.text(0.5, 0.5, 'No optimization data\navailable', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Best Hyperparameters', fontsize=14)
        
        # 6. Convergence analysis
        ax6 = axes[1, 2]
        if 'value' in tuning_results.columns:
            try:
                values = pd.to_numeric(tuning_results['value'], errors='coerce')
                valid_mask = ~values.isna()
                if valid_mask.sum() > 0:
                    valid_values = values[valid_mask]
                    valid_indices = tuning_results.index[valid_mask]
                    best_so_far = valid_values.cummax()
                    
                    ax6.plot(valid_indices, best_so_far, 'r-', linewidth=2)
                    ax6.fill_between(valid_indices, best_so_far, alpha=0.3)
                    ax6.set_title('Best Value Over Time', fontsize=14)
                    ax6.set_xlabel('Trial')
                    ax6.set_ylabel('Best Objective Value')
                    ax6.grid(True, alpha=0.3)
                else:
                    ax6.text(0.5, 0.5, 'No valid optimization\nvalues found', 
                            transform=ax6.transAxes, ha='center', va='center')
                    ax6.set_title('Best Value Over Time', fontsize=14)
            except Exception:
                ax6.text(0.5, 0.5, 'Convergence analysis\nunavailable', 
                        transform=ax6.transAxes, ha='center', va='center')
                ax6.set_title('Best Value Over Time', fontsize=14)
        else:
            ax6.text(0.5, 0.5, 'No optimization values\navailable', 
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Best Value Over Time', fontsize=14)
        
        plt.suptitle('Hyperparameter Optimization Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            filepath = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(filepath, dpi=300 if fmt == 'png' else None)
        
        plt.close()
        logger.info(f"Hyperparameter analysis plots saved to {self.output_dir}")
        
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            save_name: str = "training_history"):
        """Plot training history metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        if 'loss' in history:
            ax1 = axes[0]
            ax1.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title('Loss During Training', fontsize=14)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot other metrics
        ax2 = axes[1]
        metrics_to_plot = [k for k in history.keys() 
                          if k not in ['loss', 'val_loss'] and not k.startswith('val_')]
        
        for metric in metrics_to_plot:
            ax2.plot(history[metric], label=metric)
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax2.plot(history[val_metric], label=val_metric, linestyle='--')
        
        ax2.set_title('Metrics During Training', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Training history plot saved to {filepath}")
    
    def plot_recommendation_distribution(self, recommendations: Dict[int, List[Tuple[int, float]]],
                                       item_metadata: pd.DataFrame,
                                       save_name: str = "recommendation_distribution"):
        """Analyze and visualize recommendation distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Item popularity in recommendations
        ax1 = axes[0, 0]
        all_recommended_items = []
        for user_recs in recommendations.values():
            all_recommended_items.extend([item_id for item_id, _ in user_recs])
        
        item_counts = pd.Series(all_recommended_items).value_counts().head(20)
        ax1.barh(range(len(item_counts)), item_counts.values)
        ax1.set_yticks(range(len(item_counts)))
        
        # Get item titles if available
        if 'title' in item_metadata.columns:
            labels = [item_metadata[item_metadata['item_id'] == idx]['title'].iloc[0][:30] 
                     if idx in item_metadata['item_id'].values else str(idx)
                     for idx in item_counts.index]
            ax1.set_yticklabels(labels)
        else:
            ax1.set_yticklabels(item_counts.index)
        
        ax1.set_title('Most Frequently Recommended Items', fontsize=14)
        ax1.set_xlabel('Number of Recommendations')
        
        # 2. Genre distribution in recommendations
        ax2 = axes[0, 1]
        if 'primary_genre' in item_metadata.columns:
            recommended_genres = []
            for item_id in all_recommended_items:
                if item_id in item_metadata['item_id'].values:
                    genre = item_metadata[item_metadata['item_id'] == item_id]['primary_genre'].iloc[0]
                    recommended_genres.append(genre)
            
            genre_counts = pd.Series(recommended_genres).value_counts().head(10)
            ax2.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
            ax2.set_title('Genre Distribution in Recommendations', fontsize=14)
        
        # 3. Score distribution
        ax3 = axes[1, 0]
        all_scores = []
        for user_recs in recommendations.values():
            all_scores.extend([score for _, score in user_recs])
        
        ax3.hist(all_scores, bins=50, edgecolor='black', alpha=0.7)
        ax3.set_title('Distribution of Recommendation Scores', fontsize=14)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_scores):.3f}')
        ax3.legend()
        
        # 4. User coverage
        ax4 = axes[1, 1]
        rec_counts_per_user = [len(recs) for recs in recommendations.values()]
        ax4.hist(rec_counts_per_user, bins=20, edgecolor='black', alpha=0.7)
        ax4.set_title('Number of Recommendations per User', fontsize=14)
        ax4.set_xlabel('Number of Recommendations')
        ax4.set_ylabel('Number of Users')
        
        plt.suptitle('Recommendation Distribution Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Recommendation distribution plot saved to {filepath}")