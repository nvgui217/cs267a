# src/data/data_loader.py
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from surprise import Dataset, Reader
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import yaml
import pickle
import hashlib

logger = logging.getLogger(__name__)

class MovieLensDataLoader:
    """Optimized MovieLens data loader with caching and memory efficiency."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_path = Path(self.config['data']['movielens_path'])
        self._cache = {}
        self._cache_dir = Path("cache")
        self._cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, data_type: str) -> str:
        """Generate cache key based on data path and type."""
        path_hash = hashlib.md5(str(self.data_path).encode()).hexdigest()[:8]
        return f"{data_type}_{path_hash}"
    
    def _load_from_disk_cache(self, cache_key: str) -> Any:
        """Load data from disk cache if available."""
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            logger.info(f"Loading {cache_key} from disk cache")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_disk_cache(self, cache_key: str, data: Any):
        """Save data to disk cache."""
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved {cache_key} to disk cache")
        
    def load_ratings(self, use_cache: bool = True) -> pd.DataFrame:
        """Load ratings with caching support."""
        cache_key = self._get_cache_key("ratings")
        
        # Check memory cache
        if use_cache and cache_key in self._cache:
            logger.info("Loading ratings from memory cache")
            return self._cache[cache_key]
        
        # Check disk cache
        if use_cache:
            cached_data = self._load_from_disk_cache(cache_key)
            if cached_data is not None:
                self._cache[cache_key] = cached_data
                return cached_data
            
        logger.info("Loading MovieLens ratings from file...")
        ratings_path = self.data_path / "u.data"
        
        # Use optimized dtypes for memory efficiency
        dtypes = {
            'user_id': np.int32,
            'item_id': np.int32,
            'rating': np.float32,
            'timestamp': np.int64
        }
        
        df = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype=dtypes
        )
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = df
            self._save_to_disk_cache(cache_key, df)
            
        logger.info(f"Loaded {len(df):,} ratings from {df['user_id'].nunique():,} users")
        return df
    
    def load_movies_with_genres(self) -> pd.DataFrame:
        """Load movie information with genre processing."""
        cache_key = self._get_cache_key("movies")
        
        # Check caches
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        cached_data = self._load_from_disk_cache(cache_key)
        if cached_data is not None:
            self._cache[cache_key] = cached_data
            return cached_data
            
        logger.info("Loading movie metadata with genres...")
        
        # Genre mapping
        genre_names = ['unknown', 'Action', 'Adventure', 'Animation',
                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Column names
        columns = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        columns.extend(genre_names)
        
        # Load with proper encoding
        movies_path = self.data_path / "u.item"
        movies_df = pd.read_csv(
            movies_path, 
            sep='|', 
            names=columns, 
            encoding='latin-1',
            dtype={'item_id': np.int32}
        )
        
        # Extract genres efficiently
        genre_matrix = movies_df[genre_names].values
        movies_df['genres'] = [
            [genre_names[i] for i, val in enumerate(row) if val == 1]
            for row in genre_matrix
        ]
        movies_df['primary_genre'] = movies_df['genres'].apply(
            lambda x: x[0] if x else 'unknown'
        )
        
        # Count genres for each movie
        movies_df['genre_count'] = movies_df['genres'].apply(len)
        
        # Drop individual genre columns to save memory
        movies_df = movies_df.drop(columns=genre_names)
        
        # Cache and return
        self._cache[cache_key] = movies_df
        self._save_to_disk_cache(cache_key, movies_df)
        
        logger.info(f"Loaded metadata for {len(movies_df):,} movies")
        return movies_df
    
    def load_user_info(self) -> pd.DataFrame:
        """Load user demographic information."""
        cache_key = self._get_cache_key("users")
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        logger.info("Loading user information...")
        users_path = self.data_path / "u.user"
        
        users_df = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            dtype={'user_id': np.int32, 'age': np.int32}
        )
        
        self._cache[cache_key] = users_df
        logger.info(f"Loaded information for {len(users_df):,} users")
        return users_df
    
    def prepare_data_splits(self) -> Dict[str, Any]:
        """Prepare both explicit and implicit data with optimized splitting."""
        ratings_df = self.load_ratings()
        
        # Temporal split if configured
        if self.config['data']['temporal_split']:
            train_df, test_df = self._temporal_split(ratings_df)
        else:
            train_df, test_df = self._random_split(ratings_df)
        
        # Prepare explicit feedback data for PMF
        explicit_data = self._prepare_explicit_data(train_df, test_df)
        
        # Prepare implicit feedback data for BPR
        implicit_data = self._prepare_implicit_data(
            ratings_df, train_df, test_df,
            threshold=self.config['data']['min_rating_threshold']
        )
        
        return {
            'ratings_df': ratings_df,
            'train_df': train_df,
            'test_df': test_df,
            'explicit': explicit_data,
            'implicit': implicit_data
        }
    
    def _temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal train-test split for realistic evaluation."""
        logger.info("Performing temporal split...")
        
        test_ratio = self.config['data']['test_size']
        train_list, test_list = [], []
        
        # Split each user's interactions temporally
        for user_id, user_data in df.groupby('user_id'):
            user_data = user_data.sort_values('timestamp')
            
            # Ensure minimum interactions in training
            min_train_items = self.config['data'].get('min_train_items', 5)
            if len(user_data) <= min_train_items:
                train_list.append(user_data)
                continue
            
            # Calculate test size
            test_size = max(1, int(len(user_data) * test_ratio))
            
            # Split temporally
            train_list.append(user_data.iloc[:-test_size])
            test_list.append(user_data.iloc[-test_size:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        logger.info(f"Train size: {len(train_df):,}, Test size: {len(test_df):,}")
        return train_df, test_df
    
    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Random train-test split with stratification option."""
        from sklearn.model_selection import train_test_split
        
        # Check if stratified split is requested
        if self.config['data'].get('stratified_split', False):
            # Create stratification bins based on user activity
            user_counts = df['user_id'].value_counts()
            user_bins = pd.qcut(user_counts, q=5, duplicates='drop')
            user_strata = df['user_id'].map(user_bins)
            
            return train_test_split(
                df, 
                test_size=self.config['data']['test_size'],
                random_state=self.config['models']['pmf']['random_state'],
                stratify=user_strata
            )
        else:
            return train_test_split(
                df, 
                test_size=self.config['data']['test_size'],
                random_state=self.config['models']['pmf']['random_state']
            )
    
    def _prepare_explicit_data(self, train_df: pd.DataFrame, 
                              test_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for explicit feedback models (PMF)."""
        reader = Reader(rating_scale=(1, 5))
        
        # Create Surprise datasets
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'item_id', 'rating']], reader
        )
        trainset = train_data.build_full_trainset()
        
        # Create testset in Surprise format
        testset = list(test_df[['user_id', 'item_id', 'rating']].itertuples(index=False))
        testset = [(uid, iid, r) for uid, iid, r in testset]
        
        return {
            'trainset': trainset,
            'testset': testset,
            'train_df': train_df,
            'test_df': test_df
        }
    
    def _prepare_implicit_data(self, full_df: pd.DataFrame,
                              train_df: pd.DataFrame, 
                              test_df: pd.DataFrame,
                              threshold: float = 4.0) -> Dict[str, Any]:
        """Prepare data for implicit feedback models (BPR) with optimization."""
        logger.info(f"Converting to implicit feedback (threshold >= {threshold})...")
        
        # Get unique users and items from full dataset
        users = full_df['user_id'].unique()
        items = full_df['item_id'].unique()
        
        # Create mappings with memory-efficient dtypes
        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {i: j for j, i in enumerate(items)}
        idx_to_user = {i: u for u, i in user_to_idx.items()}
        idx_to_item = {j: i for i, j in item_to_idx.items()}
        
        # Convert to implicit feedback
        def create_sparse_matrix(df, shape):
            """Create sparse matrix efficiently."""
            df_implicit = df[df['rating'] >= threshold].copy()
            
            row = df_implicit['user_id'].map(user_to_idx).values
            col = df_implicit['item_id'].map(item_to_idx).values
            
            # Use confidence weights if configured
            if self.config['data'].get('use_confidence_weights', False):
                # Higher ratings -> higher confidence
                data = df_implicit['rating'].values.astype(np.float32)
            else:
                data = np.ones(len(df_implicit), dtype=np.float32)
            
            matrix = csr_matrix((data, (row, col)), shape=shape, dtype=np.float32)
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
    
    def create_validation_split(self, train_df: pd.DataFrame, 
                               val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create validation split from training data for hyperparameter tuning."""
        from sklearn.model_selection import train_test_split
        
        train_split, val_split = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=self.config['models']['pmf']['random_state']
        )
        
        logger.info(f"Created validation split - Train: {len(train_split):,}, Val: {len(val_split):,}")
        return train_split, val_split