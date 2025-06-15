import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualization tools for recommender systems."""
    
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_item_factors_by_genre(self, pmf_factors: np.ndarray, 
                                  bpr_factors: np.ndarray,
                                  item_ids: List[int],
                                  movies_df: pd.DataFrame,
                                  save_name: str = "item_factors_by_genre"):
        """Create item-factor scatter plot colored by genre (Medium goal requirement)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        
        # Process PMF factors
        pmf_2d = pca.fit_transform(pmf_factors)
        
        # Map item indices to genres
        genres = []
        for idx in range(len(pmf_factors)):
            item_id = item_ids[idx]
            movie = movies_df[movies_df['item_id'] == item_id]
            if not movie.empty:
                genres.append(movie.iloc[0]['primary_genre'])
            else:
                genres.append('unknown')
        
        # Get unique genres and create color mapping
        unique_genres = list(set(genres))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = {g: colors[i] for i, g in enumerate(unique_genres)}
        
        # Plot PMF
        for genre in unique_genres:
            mask = [g == genre for g in genres]
            ax1.scatter(pmf_2d[mask, 0], pmf_2d[mask, 1], 
                       c=[genre_to_color[genre]], label=genre, alpha=0.6, s=30)
        
        ax1.set_title('PMF Item Factors (PCA)', fontsize=14)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Process BPR factors
        bpr_2d = pca.fit_transform(bpr_factors)
        
        # Plot BPR
        for genre in unique_genres:
            mask = [g == genre for g in genres]
            ax2.scatter(bpr_2d[mask, 0], bpr_2d[mask, 1], 
                       c=[genre_to_color[genre]], label=genre, alpha=0.6, s=30)
        
        ax2.set_title('BPR-MF Item Factors (PCA)', fontsize=14)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                  ncol=1, title="Primary Genre")
        
        plt.tight_layout()
        
        # Save plot
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Item factor plot saved to {filepath}")
    
    def plot_model_comparison(self, results: Dict[str, Any], 
                            save_name: str = "model_comparison"):
        """Plot comparison of model performances."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract metrics
        models = ['PMF', 'BPR-MF']
        rmse_values = [results['pmf_rmse'], results['bpr_rmse_equivalent']]
        
        # Create bar plot
        bars = ax.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e'])
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        ax.set_title('Model Comparison: RMSE on Rating Prediction Task', fontsize=14)
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_ylim(0, max(rmse_values) * 1.2)
        
        # Add a note
        ax.text(0.5, 0.95, 'Note: BPR-MF RMSE is computed by converting scores to rating scale',
               transform=ax.transAxes, ha='center', va='top', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save plot
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Model comparison plot saved to {filepath}")