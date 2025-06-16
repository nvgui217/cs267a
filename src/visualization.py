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
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_item_factors_by_genre(self, pmf_factors: np.ndarray, 
                                  bpr_factors: np.ndarray,
                                  item_ids: List[int],
                                  movies_df: pd.DataFrame,
                                  save_name: str = "item_factors_by_genre"):
        """Create item-factor scatter plot colored by genre (Medium goal requirement)."""
        
        print(f"DEBUG: PMF factors shape: {pmf_factors.shape}")
        print(f"DEBUG: BPR factors shape: {bpr_factors.shape}")
        print(f"DEBUG: Movies df columns: {movies_df.columns.tolist()}")
        print(f"DEBUG: Movies df shape: {movies_df.shape}")
        
        valid_indices = []
        valid_item_ids = []
        for i, item_id in enumerate(item_ids):
            if item_id in movies_df['item_id'].values:
                valid_indices.append(i)
                valid_item_ids.append(item_id)
        
        valid_indices = np.array(valid_indices)
        pmf_factors_valid = pmf_factors[valid_indices]
        
        # Handle BPR factor dimension mismatch
        if bpr_factors.shape[1] > pmf_factors.shape[1]:
            print(f"DEBUG: Truncating BPR factors from {bpr_factors.shape[1]} to {pmf_factors.shape[1]} dimensions")
            bpr_factors = bpr_factors[:, :pmf_factors.shape[1]]  # Remove extra dimensions (likely bias)
        
        # Ensure we get the same items for BPR as PMF
        min_items = min(len(pmf_factors), len(bpr_factors))
        valid_indices_adjusted = valid_indices[valid_indices < min_items]
        
        pmf_factors_valid = pmf_factors[valid_indices_adjusted]
        bpr_factors_valid = bpr_factors[valid_indices_adjusted]
        
        print(f"DEBUG: Valid items after adjustment: {len(valid_indices_adjusted)}")
        print(f"DEBUG: PMF factors range: [{pmf_factors_valid.min():.4f}, {pmf_factors_valid.max():.4f}]")
        print(f"DEBUG: BPR factors range: [{bpr_factors_valid.min():.4f}, {bpr_factors_valid.max():.4f}]")
        print(f"DEBUG: Final shapes - PMF: {pmf_factors_valid.shape}, BPR: {bpr_factors_valid.shape}")
        
        pca = PCA(n_components=2)
        pmf_2d = pca.fit_transform(pmf_factors_valid)
        pmf_var_explained = pca.explained_variance_ratio_
        
        # FIX: Use transform, not fit_transform for BPR
        bpr_2d = pca.transform(bpr_factors_valid)
        bpr_var_explained = pmf_var_explained  # Same as PMF since we use same PCA
        
        print(f"DEBUG: PMF PCA range: x=[{pmf_2d[:, 0].min():.3f}, {pmf_2d[:, 0].max():.3f}], y=[{pmf_2d[:, 1].min():.3f}, {pmf_2d[:, 1].max():.3f}]")
        print(f"DEBUG: BPR PCA range: x=[{bpr_2d[:, 0].min():.3f}, {bpr_2d[:, 0].max():.3f}], y=[{bpr_2d[:, 1].min():.3f}, {bpr_2d[:, 1].max():.3f}]")
        
        # Check if genres column exists or if we need to parse it differently
        sample_movie = movies_df.iloc[0]
        print(f"DEBUG: Sample movie row: {sample_movie}")
        
        # Handle different genre formats
        if 'primary_genre' in movies_df.columns:
            # Single primary genre per movie
            all_genres = set(movies_df['primary_genre'].dropna().unique())
            genres = list(all_genres)[:6]  # Take first 6 genres
        elif 'genres' in movies_df.columns:
            # Genres are stored as a list or string, extract unique genres
            all_genres = set()
            for _, row in movies_df.iterrows():
                if isinstance(row['genres'], list):
                    all_genres.update(row['genres'])
                elif isinstance(row['genres'], str):
                    # Assuming pipe-separated genres like "Action|Comedy|Drama"
                    all_genres.update(row['genres'].split('|'))
            genres = list(all_genres)[:6]  # Take first 6 genres
        else:
            # Assume one-hot encoded genre columns
            possible_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 
                             'Thriller', 'Adventure', 'Animation', 'Children\'s']
            genres = [g for g in possible_genres if g in movies_df.columns][:6]
        
        print(f"DEBUG: Using genres: {genres}")
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        points_plotted = 0
        
        for idx, genre in enumerate(genres):
            mask = []
            
            # Use adjusted valid item IDs
            adjusted_valid_item_ids = [item_ids[i] for i in valid_indices_adjusted]
            
            for item_id in adjusted_valid_item_ids:
                movie_row = movies_df[movies_df['item_id'] == item_id].iloc[0]
                
                # Handle different genre formats
                if 'primary_genre' in movies_df.columns:
                    has_genre = movie_row['primary_genre'] == genre
                elif 'genres' in movies_df.columns:
                    if isinstance(movie_row['genres'], list):
                        has_genre = genre in movie_row['genres']
                    elif isinstance(movie_row['genres'], str):
                        has_genre = genre in movie_row['genres'].split('|')
                    else:
                        has_genre = False
                else:
                    # One-hot encoded
                    has_genre = movie_row.get(genre, 0) == 1
                
                mask.append(has_genre)
            
            mask = np.array(mask)
            genre_count = mask.sum()
            print(f"DEBUG: Genre '{genre}' has {genre_count} movies")
            
            if mask.any():
                ax1.scatter(pmf_2d[mask, 0], pmf_2d[mask, 1], 
                           c=[colors[idx]], label=f'{genre} ({genre_count})', alpha=0.7, s=50)
                ax2.scatter(bpr_2d[mask, 0], bpr_2d[mask, 1], 
                           c=[colors[idx]], label=f'{genre} ({genre_count})', alpha=0.7, s=50)
                points_plotted += genre_count
        
        print(f"DEBUG: Total points plotted: {points_plotted}")
        
        # If no points plotted with genres, plot all points without color coding
        if points_plotted == 0:
            print("DEBUG: No genre-based points found, plotting all points")
            ax1.scatter(pmf_2d[:, 0], pmf_2d[:, 1], alpha=0.7, s=50, c='blue', label='All items')
            ax2.scatter(bpr_2d[:, 0], bpr_2d[:, 1], alpha=0.7, s=50, c='red', label='All items')
        
        ax1.set_title('PMF Item Factors (PCA)', fontsize=14)
        ax1.set_xlabel(f'PC1 ({pmf_var_explained[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pmf_var_explained[1]:.1%} variance)')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('BPR-MF Item Factors (PCA)', fontsize=14)
        ax2.set_xlabel(f'PC1 ({bpr_var_explained[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({bpr_var_explained[1]:.1%} variance)')
        ax2.grid(True, alpha=0.3)
        
        # Only add legend if we have labeled points
        if points_plotted > 0:
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plotted {len(valid_indices_adjusted)} items with metadata out of {len(item_ids)} total items")
        logger.info(f"Item factor plot saved to {filepath}")
    
    def plot_model_comparison(self, results: Dict[str, Any], 
                            save_name: str = "model_comparison"):
        """Plot comparison of model performances."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = ['PMF', 'BPR-MF']
        rmse_values = [results['pmf_rmse'], results['bpr_rmse_equivalent']]
        
        bars = ax.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e'])
        
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        ax.set_title('Model Comparison: RMSE on Rating Prediction Task', fontsize=14)
        ax.set_ylabel('RMSE (lower is better)')
        ax.set_ylim(0, max(rmse_values) * 1.2)
        
        ax.text(0.5, 0.95, 'Note: BPR-MF RMSE is computed by converting scores to rating scale',
               transform=ax.transAxes, ha='center', va='top', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{save_name}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        logger.info(f"Model comparison plot saved to {filepath}")