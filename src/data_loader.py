import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MovieLensDataLoader:
    """Data loader for MovieLens 100K dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings from u.data file."""
        logger.info("Loading MovieLens-100K ratings...")
        ratings_path = self.data_path / "u.data"
        
        df = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': int, 'item_id': int, 'rating': float, 'timestamp': int}
        )
        
        logger.info(f"Loaded {len(df):,} ratings from {df['user_id'].nunique():,} users "
                   f"and {df['item_id'].nunique():,} items")
        return df
    
    def load_movies_with_genres(self) -> pd.DataFrame:
        """Load movie information including genres."""
        logger.info("Loading movie metadata with genres...")
        
        # Genre names as specified in MovieLens
        genre_names = ['unknown', 'Action', 'Adventure', 'Animation',
                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Column names
        columns = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        columns.extend(genre_names)
        
        # Load movies
        movies_path = self.data_path / "u.item"
        movies_df = pd.read_csv(
            movies_path, 
            sep='|', 
            names=columns, 
            encoding='latin-1'
        )
        
        # Extract primary genre (first genre if multiple, otherwise 'unknown')
        movies_df['primary_genre'] = 'unknown'
        for idx, row in movies_df.iterrows():
            for genre in genre_names:
                if row[genre] == 1:
                    movies_df.at[idx, 'primary_genre'] = genre
                    break
        
        logger.info(f"Loaded metadata for {len(movies_df):,} movies")
        return movies_df[['item_id', 'title', 'primary_genre']]
    
    def prepare_data_splits(self, test_size: float = 0.2, 
                          random_state: int = 42) -> Dict[str, Any]:
        """Prepare train-test splits for both explicit and implicit feedback."""
        ratings_df = self.load_ratings()
        
        # Create train-test split
        train_df, test_df = train_test_split(
            ratings_df, 
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Train size: {len(train_df):,}, Test size: {len(test_df):,}")
        
        # Prepare explicit feedback data for PMF
        explicit_data = self._prepare_explicit_data(train_df, test_df)
        
        # Prepare implicit feedback data for BPR
        implicit_data = self._prepare_implicit_data(
            ratings_df, train_df, test_df, threshold=3.0
        )
        
        return {
            'ratings_df': ratings_df,
            'train_df': train_df,
            'test_df': test_df,
            'explicit': explicit_data,
            'implicit': implicit_data
        }
    
    def _prepare_explicit_data(self, train_df: pd.DataFrame, 
                              test_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for explicit feedback models (PMF)."""
        reader = Reader(rating_scale=(1, 5))
        
        # Create Surprise datasets
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'item_id', 'rating']], reader
        )
        trainset = train_data.build_full_trainset()
        
        # Create testset
        testset = [(row.user_id, row.item_id, row.rating) 
                   for row in test_df.itertuples(index=False)]
        
        return {
            'trainset': trainset,
            'testset': testset,
            'train_df': train_df,
            'test_df': test_df
        }
    
    def _prepare_implicit_data(self, full_df: pd.DataFrame,
                              train_df: pd.DataFrame, 
                              test_df: pd.DataFrame,
                              threshold: float = 3.0) -> Dict[str, Any]:
        """Prepare data for implicit feedback models (BPR)."""
        logger.info(f"Converting to implicit feedback (threshold >= {threshold})...")
        
        # Get unique users and items
        users = full_df['user_id'].unique()
        items = full_df['item_id'].unique()
        
        # Create mappings
        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {i: j for j, i in enumerate(items)}
        idx_to_user = {i: u for u, i in user_to_idx.items()}
        idx_to_item = {j: i for i, j in item_to_idx.items()}
        
        # Convert to implicit feedback
        def create_sparse_matrix(df, shape):
            df_implicit = df[df['rating'] >= threshold].copy()
            
            row = df_implicit['user_id'].map(user_to_idx).values
            col = df_implicit['item_id'].map(item_to_idx).values
            data = np.ones(len(df_implicit))
            
            matrix = csr_matrix((data, (row, col)), shape=shape)
            return matrix
        
        shape = (len(users), len(items))
        train_matrix = create_sparse_matrix(train_df, shape)
        test_matrix = create_sparse_matrix(test_df, shape)
        
        logger.info(f"Implicit matrices created - Shape: {shape}, "
                   f"Train interactions: {train_matrix.nnz:,}, "
                   f"Test interactions: {test_matrix.nnz:,}")
        
        return {
            'train_matrix': train_matrix,
            'test_matrix': test_matrix,
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item,
            'shape': shape
        }