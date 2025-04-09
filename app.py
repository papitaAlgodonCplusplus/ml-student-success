from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

os.makedirs('model', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Try to load the model parameters from JSON if it exists
model_params = None
try:
    with open("model/model_params.json", "rb") as f:
        model_params = json.load(f)
    print("Model parameters loaded successfully!")
except FileNotFoundError:
    print("Model parameters file not found. The prediction will use a simulated model.")

# Define feature importance for explanation
feature_importance = {
    "years_of_experience": 0.25,
    "technical_skills": 0.22,
    "certificates": 0.15,
    "personal_projects": 0.14,
    "academic_achievements": 0.10,
    "extra_academic_achievements": 0.08,
    "networking_referees": 0.06
}

# If model_params exists, update feature_importance with actual values
if model_params and 'feature_importances' in model_params:
    feature_names = [
        "years_of_experience", "technical_skills", "certificates",
        "personal_projects", "academic_achievements", "extra_academic_achievements",
        "networking_referees"
    ]
    
    # Update feature importance with values from the model
    importances = model_params['feature_importances']
    if len(importances) == len(feature_names):
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = importances[i]

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Always use simulation with updated feature importance
        prediction_probability = predict_selection(data)
        prediction = 1 if prediction_probability > 0.5 else 0
        
        # Generate personalized recommendations based on input
        recommendations = generate_recommendations(data, prediction)
        
        # Prepare feature importance for this prediction
        explanation = {}
        for feature, importance in feature_importance.items():
            if feature in data:
                explanation[feature] = {
                    "value": data[feature],
                    "importance": importance,
                    "impact": calculate_feature_impact(feature, data[feature], importance)
                }
        
        # Get selection likelihood based on probability
        selection_likelihood = get_selection_likelihood(prediction_probability)
        
        response = {
            "prediction": int(prediction),
            "selection_probability": float(prediction_probability),
            "selection_likelihood": selection_likelihood,
            "recommendations": recommendations,
            "explanation": explanation
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def predict_selection(data):
    selection_prob = 0
    
    for feature, importance in feature_importance.items():
        if feature in data:
            if feature == "years_of_experience":
                normalized_value = min(data[feature], 15) / 15  # Cap at 15 years
            elif feature == "technical_skills":
                normalized_value = min(data[feature], 10) / 10  # Cap at 10 skills
            elif feature == "certificates":
                normalized_value = min(data[feature], 5) / 5  # Cap at 5 certificates
            elif feature == "personal_projects":
                normalized_value = min(data[feature], 8) / 8  # Cap at 8 projects
            elif feature == "academic_achievements":
                normalized_value = data[feature] / 100  # Percentage-based
            elif feature == "extra_academic_achievements":
                normalized_value = min(data[feature], 5) / 5  # Cap at 5 achievements
            elif feature == "networking_referees":
                normalized_value = min(data[feature], 5) / 5  # Cap at 5 referees
            else:
                normalized_value = 0.5  # Default
            
            # Add weighted contribution
            selection_prob += normalized_value * importance
    
    # Add some randomness to simulate model uncertainty
    selection_prob = min(max(selection_prob + (np.random.random() * 0.1 - 0.05), 0), 1)
    
    return selection_prob

def calculate_feature_impact(feature, value, importance):
    # Simplified impact calculation - would be more sophisticated in real app
    feature_ranges = {
        "years_of_experience": {"min": 0, "max": 15, "optimal": 8},
        "technical_skills": {"min": 0, "max": 10, "optimal": 7},
        "certificates": {"min": 0, "max": 5, "optimal": 3},
        "personal_projects": {"min": 0, "max": 8, "optimal": 5},
        "academic_achievements": {"min": 0, "max": 100, "optimal": 85},
        "extra_academic_achievements": {"min": 0, "max": 5, "optimal": 3},
        "networking_referees": {"min": 0, "max": 5, "optimal": 4}
    }
    
    if feature in feature_ranges:
        range_info = feature_ranges[feature]
        normalized_value = (value - range_info["min"]) / (range_info["max"] - range_info["min"])
        normalized_optimal = (range_info["optimal"] - range_info["min"]) / (range_info["max"] - range_info["min"])
        
        # Distance from optimal
        distance = abs(normalized_value - normalized_optimal)
        impact = (1 - distance) * importance
        return float(impact)
    
    return 0.0

def get_selection_likelihood(probability):
    if probability >= 0.8:
        return "Very Likely"
    elif probability >= 0.6:
        return "Likely"
    elif probability >= 0.4:
        return "Possible"
    else:
        return "Unlikely"

def generate_recommendations(data, prediction):
    recommendations = []
    
    # Example recommendation logic
    if data.get("years_of_experience", 0) < 2:
        recommendations.append({
            "area": "Work Experience",
            "suggestion": "Highlight internships or relevant project experience to compensate for limited professional experience",
            "impact": "high"
        })
    
    if data.get("technical_skills", 0) < 5:
        recommendations.append({
            "area": "Technical Skills",
            "suggestion": "Expand your technical toolkit by learning in-demand programming languages or frameworks",
            "impact": "high" 
        })
    
    if data.get("certificates", 0) < 2:
        recommendations.append({
            "area": "Professional Certifications",
            "suggestion": "Pursue relevant industry certifications to validate your expertise",
            "impact": "medium"
        })
        
    if data.get("personal_projects", 0) < 3:
        recommendations.append({
            "area": "Portfolio Development",
            "suggestion": "Create personal or open-source projects that showcase your problem-solving abilities",
            "impact": "medium"
        })
    
    if data.get("networking_referees", 0) < 2:
        recommendations.append({
            "area": "Professional Network",
            "suggestion": "Expand your industry connections through meetups, conferences, or online communities",
            "impact": "low"
        })
    
    return recommendations

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)