import pandas as pd
import numpy as np
from load_fitness_tracker_dataset import FitnessTrackerLoader

class KaggleDatasetPreprocessor:
    """Convert Kaggle dataset to preexisting training data"""
    
    def __init__(self):
        self.loader = FitnessTrackerLoader()
        self.training_data = None
    
    def prepare_foundation_data(self):
        """Prepare data for foundation model training"""
        self.training_data = self.loader.create_training_data()
        
        # Split by difficulty
        easy_data = self.training_data[self.training_data['difficulty'] == 0]
        medium_data = self.training_data[self.training_data['difficulty'] == 1]
        hard_data = self.training_data[self.training_data['difficulty'] == 2]
        
        # Convert to tuples for PREEXISTING_DATA format
        easy_tuples = [
            (row['form_score'], row['heart_rate'], row['spo2'], row['stress'], 
             row['completion'], row['form_trend'])
            for _, row in easy_data.iterrows()
        ]
        
        medium_tuples = [
            (row['form_score'], row['heart_rate'], row['spo2'], row['stress'], 
             row['completion'], row['form_trend'])
            for _, row in medium_data.iterrows()
        ]
        
        hard_tuples = [
            (row['form_score'], row['heart_rate'], row['spo2'], row['stress'], 
             row['completion'], row['form_trend'])
            for _, row in hard_data.iterrows()
        ]
        
        return {
            'easy': easy_tuples,
            'medium': medium_tuples,
            'hard': hard_tuples,
        }
    
    def get_numpy_arrays(self):
        """Get training arrays"""
        return self.loader.get_numpy_arrays(self.training_data)

def get_preexisting_training_data():
    """Load Kaggle data as preexisting foundation"""
    processor = KaggleDatasetPreprocessor()
    processor.training_data = processor.loader.create_training_data()
    
    X = processor.training_data[['form_score', 'heart_rate', 'spo2', 'stress', 'completion', 'form_trend']].values
    y = processor.training_data['difficulty'].values
    
    return np.array(X), np.array(y)

def get_preexisting_data_by_difficulty(difficulty):
    """Get preexisting data for specific difficulty"""
    processor = KaggleDatasetPreprocessor()
    processor.training_data = processor.loader.create_training_data()
    
    if difficulty == 'easy':
        data = processor.training_data[processor.training_data['difficulty'] == 0]
    elif difficulty == 'medium':
        data = processor.training_data[processor.training_data['difficulty'] == 1]
    else:
        data = processor.training_data[processor.training_data['difficulty'] == 2]
    
    X = data[['form_score', 'heart_rate', 'spo2', 'stress', 'completion', 'form_trend']].values
    
    return np.array(X)

if __name__ == '__main__':
    processor = KaggleDatasetPreprocessor()
    foundation = processor.prepare_foundation_data()
    
    print('✅ Foundation data prepared:')
    print(f'   Easy: {len(foundation["easy"])} samples')
    print(f'   Medium: {len(foundation["medium"])} samples')
    print(f'   Hard: {len(foundation["hard"])} samples')
    
    X, y, df = processor.get_numpy_arrays()
    print(f'\n✅ NumPy arrays:')
    print(f'   X shape: {X.shape}')
    print(f'   y shape: {y.shape}')