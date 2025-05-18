from flask import Flask, render_template, request, jsonify
import pickle
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib

load_dotenv()

app = Flask(__name__)

# Load the model and label encoder
try:
    with open('player_position_model.pkl', 'rb') as f:
        model = joblib.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = joblib.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Football API configuration
FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', '4849a949ed194c8d9bd74cfb78d6e61e')  # Get from football-data.org
FOOTBALL_API_BASE = 'http://api.football-data.org/v4'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data with default values
            stats = {
                'age': int(request.form.get('age', 23)),
                'height_cm': int(request.form.get('height_cm', 180)),
                'weight_kgs': int(request.form.get('weight_kgs', 74)),
                'overall_rating': float(request.form.get('overall_rating', 90)),
                'potential': float(request.form.get('potential', 95)),
                'sprint_speed': float(request.form.get('sprint_speed', 42)),
                'short_passing': float(request.form.get('short_passing', 75)),
                'long_passing': float(request.form.get('long_passing', 77)),
                'dribbling': float(request.form.get('dribbling', 90)),
                'strength': float(request.form.get('strength', 70))
            }
            
            # Print debug information
            print("Received stats:", stats)
            
            # Make prediction using all features in the correct order
            features = [
                stats['age'],
                stats['height_cm'],
                stats['weight_kgs'],
                stats['overall_rating'],
                stats['potential'],
                stats['sprint_speed'],
                stats['short_passing'],
                stats['long_passing'],
                stats['dribbling'],
                stats['strength']
            ]
            
            # Print debug information
            print("Features array:", features)
            print("Number of features:", len(features))
            
            prediction = model.predict([features])
            position = label_encoder.inverse_transform(prediction)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba([features])[0]
            confidence = max(probabilities) * 100
            
            return render_template('predict.html', 
                                 stats=stats,
                                 position=position,
                                 confidence=confidence,
                                 probabilities=probabilities.tolist())
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/wiki')
def wiki():
    player_name = request.args.get('player', '')
    if not player_name:
        return render_template('wiki.html')
    
    # Wikipedia API call
    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{player_name}"
    response = requests.get(wiki_url)
    
    if response.status_code == 200:
        data = response.json()
        return render_template('wiki.html',
                             player_name=player_name,
                             summary=data.get('extract', 'No summary available'),
                             image_url=data.get('thumbnail', {}).get('source', None))
    
    return render_template('wiki.html', error="Player not found")

@app.route('/live')
def live():
    headers = {'X-Auth-Token': FOOTBALL_API_KEY}
    
    # Try to get live matches
    live_url = f"{FOOTBALL_API_BASE}/matches"
    response = requests.get(live_url, headers=headers)
    
    if response.status_code == 200:
        matches = response.json().get('matches', [])
        if matches:
            return render_template('live.html', matches=matches, is_live=True)
    
    # If no live matches, get last week's matches
    last_week = datetime.now() - timedelta(days=7)
    past_url = f"{FOOTBALL_API_BASE}/matches?dateFrom={last_week.strftime('%Y-%m-%d')}&dateTo={datetime.now().strftime('%Y-%m-%d')}"
    response = requests.get(past_url, headers=headers)
    
    if response.status_code == 200:
        matches = response.json().get('matches', [])
        return render_template('live.html', matches=matches, is_live=False)
    
    return render_template('live.html', error="Could not fetch match data")

if __name__ == '__main__':
    app.run(debug=True) 