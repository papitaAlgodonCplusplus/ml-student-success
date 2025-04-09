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
    "previous_gpa": 0.23,
    "assignment_completion": 0.21,
    "study_hours_weekly": 0.18,
    "submission_timeliness": 0.15,
    "debugging_time": 0.12,
    "forum_participation": 0.07,
    "office_hours_attendance": 0.04
}

# If model_params exists, update feature_importance with actual values
if model_params and 'feature_importances' in model_params:
    feature_names = [
        "previous_gpa", "assignment_completion", "study_hours_weekly",
        "submission_timeliness", "debugging_time", "forum_participation",
        "office_hours_attendance"
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
        prediction_probability = predict(data)
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
        
        # Get risk level based on probability
        risk_level = get_risk_level(prediction_probability)
        
        response = {
            "prediction": int(prediction),
            "success_probability": float(prediction_probability),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "explanation": explanation
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

def predict(data):
    success_prob = 0
    
    for feature, importance in feature_importance.items():
        if feature in data:
            if feature == "previous_gpa":
                normalized_value = data[feature] / 4.0
            elif feature in ["assignment_completion", "submission_timeliness", "forum_participation"]:
                normalized_value = data[feature] / 100
            elif feature == "study_hours_weekly":
                normalized_value = min(data[feature], 20) / 20
            elif feature == "debugging_time":
                normalized_value = 1 - min(data[feature], 20) / 20
            elif feature == "office_hours_attendance":
                normalized_value = min(data[feature], 10) / 10
            else:
                normalized_value = 0.5  # Default
            
            # Add weighted contribution
            success_prob += normalized_value * importance
    
    # Add some randomness to simulate model uncertainty
    success_prob = min(max(success_prob + (np.random.random() * 0.1 - 0.05), 0), 1)
    
    return success_prob

def calculate_feature_impact(feature, value, importance):
    # Simplified impact calculation - would be more sophisticated in real app
    feature_ranges = {
        "previous_gpa": {"min": 0, "max": 4.0, "optimal": 3.5},
        "assignment_completion": {"min": 0, "max": 100, "optimal": 95},
        "study_hours_weekly": {"min": 0, "max": 30, "optimal": 15},
        "submission_timeliness": {"min": 0, "max": 100, "optimal": 90},
        "debugging_time": {"min": 0, "max": 20, "optimal": 7},
        "forum_participation": {"min": 0, "max": 100, "optimal": 70},
        "office_hours_attendance": {"min": 0, "max": 10, "optimal": 5}
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

def get_risk_level(probability):
    if probability >= 0.8:
        return "Low Risk"
    elif probability >= 0.6:
        return "Moderate Risk"
    elif probability >= 0.4:
        return "Elevated Risk"
    else:
        return "High Risk"

def generate_recommendations(data, prediction):
    recommendations = []
    
    # Example recommendation logic
    if data.get("assignment_completion", 100) < 90:
        recommendations.append({
            "area": "Assignment Completion",
            "suggestion": "Try to complete all assignments, even with partial solutions",
            "impact": "high"
        })
    
    if data.get("study_hours_weekly", 0) < 10:
        recommendations.append({
            "area": "Study Time",
            "suggestion": "Increase study time by at least 2 hours per week",
            "impact": "medium" 
        })
    
    if data.get("debugging_time", 0) > 15:
        recommendations.append({
            "area": "Debugging Strategy",
            "suggestion": "Consider using rubber duck debugging and divide complex problems",
            "impact": "medium"
        })
        
    if data.get("forum_participation", 0) < 50:
        recommendations.append({
            "area": "Forum Engagement",
            "suggestion": "Actively participate in discussions and ask questions",
            "impact": "low"
        })
    
    if data.get("submission_timeliness", 0) < 80:
        recommendations.append({
            "area": "Time Management",
            "suggestion": "Start assignments earlier to allow buffer time",
            "impact": "high"
        })
    
    return recommendations

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)