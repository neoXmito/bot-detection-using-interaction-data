from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import random
import time
import pandas as pd
from sqlalchemy import func, case

import joblib 


#initialize the model
model = joblib.load('models\\best_model_random_forest.pkl')
# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deployement.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Database models
class CaptchaAttempt(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    correct_answer = db.Column(db.String(50))
    user_answer = db.Column(db.String(50))
    success = db.Column(db.Boolean)
    start_time = db.Column(db.Float)
    end_time = db.Column(db.Float)
    interactions = db.relationship('Interaction', backref='attempt', lazy=True)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.String(20), db.ForeignKey('captcha_attempt.id'))
    type = db.Column(db.String(20))
    x = db.Column(db.Float)
    y = db.Column(db.Float)
    key = db.Column(db.String(10))
    speed = db.Column(db.Float)
    timestamp = db.Column(db.Float)
    extra_data = db.Column(db.JSON)

# Configuration
CAPTCHA_IMAGES_DIR = 'captcha_images'
current_captchas = {}  # Stores {captcha_id: answer}



def create_dataset():
    with app.app_context():
        # Query base attempt data
        attempts = CaptchaAttempt.query.all()
        
        dataset = []
        
        for attempt in attempts:
            # Get all interactions for this attempt
            interactions = Interaction.query.filter_by(attempt_id=attempt.id).all()
            
            # Calculate interaction metrics
            mouse_events = [i for i in interactions if i.type == 'mousemove']
            click_events = [i for i in interactions if i.type == 'click']
            key_events = [i for i in interactions if i.type == 'keydown']
            input_changes = [i for i in interactions if i.type == 'input_change']
            
            # Calculate mouse movement statistics
            mouse_speeds = [i.speed for i in mouse_events if i.speed is not None]
            
            # Calculate time-based features
            total_time = attempt.end_time - attempt.start_time
            
            # Create feature dictionary
            features = {
                'success': 1 if attempt.success else 0,
                'total_time': total_time,
                
                # Mouse metrics
                'mouse_move_count': len(mouse_events),
                'avg_mouse_speed': sum(mouse_speeds)/len(mouse_speeds) if mouse_speeds else 0,
                'max_mouse_speed': max(mouse_speeds) if mouse_speeds else 0,
                
                # Input metrics
                'input_change_count': len(input_changes),
                'final_input_length': len(attempt.user_answer),
                'edit_distance': levenshtein(attempt.correct_answer, attempt.user_answer),
                
                # Temporal features
                'time_per_character': total_time / len(attempt.user_answer) if attempt.user_answer else 0,
                
                # Additional features
                'error_rate': len(input_changes) / len(attempt.user_answer) if attempt.user_answer else 0
            }
            
            dataset.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)

        #only get the last row in the df
        df = df.tail(1)
        
        # Save to CSV
        df.to_csv('dataset\captcha_dataset.csv', index=False)
        return df

def levenshtein(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def get_random_captcha():
    """Get random CAPTCHA image and create new entry"""
    images = os.listdir(CAPTCHA_IMAGES_DIR)
    if not images:
        return None, None
    selected = random.choice(images)
    captcha_id = str(int(time.time() * 1000))  # High precision ID
    answer = os.path.splitext(selected)[0]
    current_captchas[captcha_id] = answer
    return captcha_id, selected

@app.route('/')
def index():
    """Main page with CAPTCHA"""
    captcha_id, filename = get_random_captcha()
    if not filename:
        return "No CAPTCHA images found", 404
    return render_template('index.html',
                         captcha_id=captcha_id,
                         captcha_image=filename)

@app.route('/captcha_images/<path:filename>')
def serve_captcha(filename):
    """Serve CAPTCHA images"""
    return send_from_directory(CAPTCHA_IMAGES_DIR, filename)

@app.route('/verify', methods=['POST'])
def verify():
    """Verify CAPTCHA answer and predict using ML model"""
    data = request.json
    captcha_id = data.get('captcha_id')
    user_answer = data.get('answer', '').strip().lower()
    
    # Get stored answer and determine verification success
    correct_answer = current_captchas.get(captcha_id, '').lower()
    success = user_answer == correct_answer
    
    # Create database records
    attempt = CaptchaAttempt(
        id=captcha_id,
        correct_answer=correct_answer,
        user_answer=user_answer,
        success=success,
        start_time=data.get('start_time'),
        end_time=data.get('end_time')
    )
    
    interactions = []
    for interaction in data.get('interactions', []):
        interactions.append(Interaction(
            attempt_id=captcha_id,
            type=interaction.get('type'),
            x=interaction.get('x'),
            y=interaction.get('y'),
            key=interaction.get('key'),
            speed=interaction.get('speed'),
            timestamp=interaction.get('timestamp'),
            extra_data=interaction.get('extra_data')
        ))
    
    db.session.add(attempt)
    db.session.add_all(interactions)
    db.session.commit()
    
    # Remove used CAPTCHA from current list
    if captcha_id in current_captchas:
        del current_captchas[captcha_id]
    
    # Create dataset (features from latest attempt) and predict class
    features_df = create_dataset()  # returns last row of feature DataFrame
    predicted_class = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    # Convert probabilities to percentages
    probabilities_dict = {
        0: round(probabilities[0] * 100, 2),
        1: round(probabilities[1] * 100, 2),
        2: round(probabilities[2] * 100, 2)
    }
    
    return jsonify({
        'success': success,
        'predicted_class': str(predicted_class),
        'probabilities': probabilities_dict
    })


@app.route('/get_new_captcha')
def new_captcha():
    """Get new CAPTCHA for client-side refresh"""
    captcha_id, filename = get_random_captcha()
    if not filename:
        return jsonify(error="No CAPTCHA available"), 404
    
    return jsonify({
        'captcha_id': captcha_id,
        'image_url': f"/captcha_images/{filename}"
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)