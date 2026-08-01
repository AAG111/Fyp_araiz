import random
from datetime import datetime

CHALLENGES = {
    'easy': [
        {'id': 'sq10_e', 'name': 'Squat Starter', 'desc': '1 set of 10 squats', 'reps': 10, 'form': 70},
        {'id': 'pu5_e', 'name': 'Push-up Start', 'desc': '1 set of 5 push-ups', 'reps': 5, 'form': 60},
        {'id': 'walk_e', 'name': 'Light Walk', 'desc': 'Walk for 60 seconds', 'reps': 1, 'form': 50},
        {'id': 'breathe_e', 'name': 'Breathe Slowly', 'desc': 'Deep breathing for 30s', 'reps': 1, 'form': 50},
    ],
    'medium': [
        {'id': 'sq15_m', 'name': 'Squat Warrior', 'desc': '2 sets of 10 squats, form ≥70%', 'reps': 10, 'form': 70, 'sets': 2},
        {'id': 'pu15_m', 'name': 'Push-up Challenge', 'desc': '2 sets of 10 push-ups', 'reps': 10, 'form': 70, 'sets': 2},
        {'id': 'dl_m', 'name': 'Deadlift Strong', 'desc': '2 sets of 8 deadlifts', 'reps': 8, 'form': 75, 'sets': 2},
        {'id': 'calm_m', 'name': 'Calm Mind', 'desc': 'Reduce stress <40% for 60s', 'reps': 1, 'form': 40},
    ],
    'hard': [
        {'id': 'sq20_h', 'name': 'Squat Master', 'desc': '3 sets of 20 squats, form ≥85%', 'reps': 20, 'form': 85, 'sets': 3},
        {'id': 'pu25_h', 'name': 'Push-up Power', 'desc': '3 sets of 25 push-ups', 'reps': 25, 'form': 80, 'sets': 3},
        {'id': 'circuit_h', 'name': 'Full Circuit', 'desc': '50 reps mixed exercises', 'reps': 50, 'form': 80},
        {'id': 'zen_h', 'name': 'Zen Master', 'desc': 'Stress <20% for 90s', 'reps': 1, 'form': 20},
    ]
}

def get_daily_challenges(difficulty='medium'):
    """Select 5 daily challenges by difficulty"""
    pool = CHALLENGES.get(difficulty, CHALLENGES['medium'])
    selected = random.sample(pool, min(5, len(pool)))
    return {
        'difficulty': difficulty,
        'challenges': selected,
        'timestamp': datetime.now().isoformat()
    }

def predict_difficulty(form_score, heart_rate, spo2, stress):
    """Simple rule-based difficulty prediction (before ML model is trained)"""
    if form_score < 70 or stress > 70:
        return 'easy'
    elif form_score < 85 and heart_rate < 140:
        return 'medium'
    else:
        return 'hard'

if __name__ == '__main__':
    print('Easy challenges:')
    for c in CHALLENGES['easy']:
        print(f"  - {c['name']}: {c['desc']}")