from database import init_db, get_challenge_logs
from adaptive_model_kaggle import AdaptiveModel
import numpy as np

def train_from_logs():
    """Train model from existing challenge logs"""
    init_db()
    model = AdaptiveModel()
    
    USER_ID = 1
    logs = get_challenge_logs(USER_ID, limit=100)
    
    if len(logs) < 10:
        print(f'⚠️ Only {len(logs)} logs — need at least 10 to train')
        return
    
    X = []
    y = []
    
    for log in logs:
        form_score = log[5] or 80
        actual_reps = log[4] or 0
        
        X.append([form_score, 70, 95, 50, min(actual_reps/10, 1.0), 1.0 if form_score > 80 else 0.5])
        y.append(0 if form_score < 70 else 1 if form_score < 85 else 2)
    
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    print(f'Training on {len(X)} samples...')
    model.train(X_arr, y_arr)
    print('✅ Model trained and saved')

if __name__ == '__main__':
    train_from_logs()