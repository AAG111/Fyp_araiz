import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime, timedelta
import json

# Use Kaggle data as foundation
try:
    from preexisting_database_from_kaggle import get_preexisting_training_data
except:
    from preexisting_database import get_preexisting_training_data

MODEL_PATH = 'adaptive_model.h5'
SCALER_PATH = 'scaler.pkl'
MODEL_META_PATH = 'model_metadata.json'

class AdaptiveModelKaggle:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = {
            'foundation_trained': False,
            'user_data_days': 0,
            'last_retrain': None,
            'last_weekly_retrain': None,
            'total_user_samples': 0,
            'model_version': 1,
            'use_user_data': False,
            'user_data_confidence': 0.0,
            'dataset': 'kaggle_fitness_tracker',
            'foundation_samples': 0,
        }
        self.load_or_create()
    
    def load_or_create(self):
        """Load existing model or create + train new one"""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            print('📦 Loading existing model...')
            self.model = keras.models.load_model(MODEL_PATH)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            if os.path.exists(MODEL_META_PATH):
                with open(MODEL_META_PATH, 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"   Foundation trained: {self.metadata['foundation_trained']}")
            print(f"   Dataset: {self.metadata['dataset']}")
            print(f"   Foundation samples: {self.metadata['foundation_samples']}")
            print(f"   User data days: {self.metadata['user_data_days']}")
            print(f"   User data confidence: {self.metadata['user_data_confidence']:.1%}")
        else:
            print('🆕 Creating and training new model with Kaggle Foundation data...')
            self._create_model()
            self.scaler = StandardScaler()
            self._train_foundation()
    
    def _create_model(self):
        """Build neural network"""
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(6,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print('✅ Model architecture created')
    
    def _train_foundation(self):
        """Train on Kaggle foundation data"""
        print('📚 Training on Kaggle Fitness Tracker dataset...')
        
        X, y = get_preexisting_training_data()
        self.metadata['foundation_samples'] = len(X)
        
        print(f'   Samples: {len(X)}')
        print(f'   Distribution - Easy: {sum(y==0)}, Medium: {sum(y==1)}, Hard: {sum(y==2)}')
        
        X_scaled = self.scaler.fit_transform(X)
        
        history = self.model.fit(
            X_scaled, y,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        self.model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        accuracy = history.history['accuracy'][-1]
        self.metadata['foundation_trained'] = True
        self.metadata['last_retrain'] = datetime.now().isoformat()
        self.metadata['last_weekly_retrain'] = datetime.now().isoformat()
        self.metadata['model_version'] = 1
        self._save_metadata()
        
        print(f'\n✅ Foundation model trained')
        print(f'   Accuracy: {accuracy:.2%}')
        print(f'   Epochs: {len(history.history["loss"])}')
        print(f'\n💡 Model ready. Using Kaggle data until 10 days of user data collected.')
    
    def update_user_data_progress(self, user_sessions):
        """Track user data progress"""
        if not user_sessions:
            self.metadata['user_data_days'] = 0
            self.metadata['use_user_data'] = False
            self.metadata['user_data_confidence'] = 0.0
            return
        
        dates = set()
        for session in user_sessions:
            dates.add(session[-1][:10] if isinstance(session[-1], str) else 'unknown')
        
        days_active = len(dates)
        self.metadata['user_data_days'] = days_active
        self.metadata['total_user_samples'] = len(user_sessions)
        
        if days_active >= 10:
            self.metadata['use_user_data'] = True
            days_after_threshold = days_active - 10
            confidence = min(0.5 + (days_after_threshold / 100), 1.0)
            self.metadata['user_data_confidence'] = confidence
        else:
            self.metadata['use_user_data'] = False
            self.metadata['user_data_confidence'] = 0.0
        
        self._save_metadata()
    
    def prepare_mixed_data(self, user_sessions):
        """Blend Kaggle + user data"""
        X_foundation, y_foundation = get_preexisting_training_data()
        X_user, y_user = self._prepare_user_data(user_sessions)
        
        if len(X_user) == 0:
            return X_foundation, y_foundation
        
        confidence = self.metadata['user_data_confidence']
        foundation_weight = 1.0 - confidence
        
        repetitions = max(1, int(foundation_weight * 5))
        X_foundation_weighted = np.repeat(X_foundation, repetitions, axis=0)
        y_foundation_weighted = np.repeat(y_foundation, repetitions)
        
        X_combined = np.vstack([X_foundation_weighted, X_user])
        y_combined = np.hstack([y_foundation_weighted, y_user])
        
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        
        print(f'📊 Mixed data: {len(X_foundation_weighted)} Kaggle + {len(X_user)} user')
        print(f'   Foundation weight: {foundation_weight:.1%}')
        print(f'   User data confidence: {confidence:.1%}')
        
        return X_combined, y_combined
    
    def _prepare_user_data(self, user_sessions):
        """Convert user sessions to training format"""
        X = []
        y = []
        
        for session in user_sessions:
            form_score = session[4] or 80
            heart_rate = session[5] or 70
            spo2 = session[6] or 95
            stress = session[7] or 50
            reps = session[3] or 0
            
            completion = min(reps / 10, 1.0)
            form_trend = 1.0 if form_score > 80 else 0.5 if form_score > 60 else 0.2
            
            X.append([form_score, heart_rate, spo2, stress, completion, form_trend])
            
            if form_score < 70 or stress > 70:
                y.append(0)
            elif form_score < 85 or heart_rate > 140:
                y.append(1)
            else:
                y.append(2)
        
        return np.array(X) if X else np.array([]).reshape(0, 6), np.array(y)
    
    def should_retrain(self):
        """Check if weekly retrain needed"""
        last_retrain = self.metadata.get('last_weekly_retrain')
        
        if not last_retrain:
            return True
        
        last_retrain_time = datetime.fromisoformat(last_retrain)
        days_since = (datetime.now() - last_retrain_time).days
        
        return days_since >= 7
    
    def train(self, user_sessions):
        """Train on mixed data"""
        if len(user_sessions) < 5:
            print(f'⚠️  Only {len(user_sessions)} user samples — need at least 5')
            return False
        
        self.update_user_data_progress(user_sessions)
        
        X_combined, y_combined = self.prepare_mixed_data(user_sessions)
        
        X_scaled = self.scaler.fit_transform(X_combined)
        
        callback = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_scaled, y_combined,
            epochs=50,
            batch_size=8,
            callbacks=[callback],
            verbose=0
        )
        
        self.model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        accuracy = history.history['accuracy'][-1]
        self.metadata['last_retrain'] = datetime.now().isoformat()
        
        if self.should_retrain():
            self.metadata['last_weekly_retrain'] = datetime.now().isoformat()
            self.metadata['model_version'] += 1
            print(f'📦 Model version: {self.metadata["model_version"]}')
        
        self._save_metadata()
        
        print(f'✅ Model retrained. Accuracy: {accuracy:.2%}')
        print(f'   User data days: {self.metadata["user_data_days"]}')
        print(f'   Using user data: {self.metadata["use_user_data"]}')
        
        return True
    
    def predict_difficulty(self, form_score, heart_rate, spo2, stress, completion, form_trend):
        """Predict difficulty"""
        X = np.array([[form_score, heart_rate, spo2, stress, completion, form_trend]])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled, verbose=0)[0]
        difficulty_idx = np.argmax(probs)
        difficulties = ['easy', 'medium', 'hard']
        
        return difficulties[difficulty_idx]
    
    def get_difficulty_probabilities(self, form_score, heart_rate, spo2, stress, completion, form_trend):
        """Get probability distribution"""
        X = np.array([[form_score, heart_rate, spo2, stress, completion, form_trend]])
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled, verbose=0)[0]
        
        return {
            'easy': float(probs[0]),
            'medium': float(probs[1]),
            'hard': float(probs[2])
        }
    
    def _save_metadata(self):
        """Save metadata"""
        with open(MODEL_META_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_status(self):
        """Get model status"""
        return {
            'model_version': self.metadata['model_version'],
            'foundation_trained': self.metadata['foundation_trained'],
            'dataset': self.metadata['dataset'],
            'foundation_samples': self.metadata['foundation_samples'],
            'user_data_days': self.metadata['user_data_days'],
            'using_user_data': self.metadata['use_user_data'],
            'user_data_confidence': f"{self.metadata['user_data_confidence']:.1%}",
            'last_retrain': self.metadata['last_retrain'],
            'days_to_next_retrain': self._days_to_next_retrain(),
            'total_user_samples': self.metadata['total_user_samples'],
        }
    
    def _days_to_next_retrain(self):
        """Days until next weekly retrain"""
        last_retrain = self.metadata.get('last_weekly_retrain')
        if not last_retrain:
            return 0
        
        last_retrain_time = datetime.fromisoformat(last_retrain)
        days_since = (datetime.now() - last_retrain_time).days
        days_left = max(0, 7 - days_since)
        
        return days_left

if __name__ == '__main__':
    model = AdaptiveModelKaggle()
    print('✅ Adaptive model ready')
    print(f'   Status: {model.get_status()}')