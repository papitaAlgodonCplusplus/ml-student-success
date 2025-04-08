from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

os.makedirs('model', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Try to load the model if it exists
model = None
scaler = None
try:
    with open("model/student_success_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. The prediction will use a simulated model.")

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
        
        if model is not None and scaler is not None:
            # Create dataframe from input
            input_df = pd.DataFrame([data])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction using the loaded model
            prediction = model.predict(input_scaled)[0]
            prediction_probability = model.predict_proba(input_scaled)[0][1]  # Get probability of positive class
        else:
            # Simulate prediction (for demo purposes when no model is available)
            prediction_probability = simulate_prediction(data)
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

def simulate_prediction(data):
    """Simulate a prediction when no model is available"""
    # This is a simplified simulation - in reality your ML model would do this
    success_prob = 0
    
    # Weight factors based on importance
    success_prob += (data.get("previous_gpa", 0) / 4.0) * 0.23
    success_prob += (data.get("assignment_completion", 0) / 100) * 0.21
    success_prob += (min(data.get("study_hours_weekly", 0), 20) / 20) * 0.18
    success_prob += (data.get("submission_timeliness", 0) / 100) * 0.15
    success_prob += (1 - min(data.get("debugging_time", 0), 20) / 20) * 0.12  # Less debugging time is better
    success_prob += (data.get("forum_participation", 0) / 100) * 0.07
    success_prob += (min(data.get("office_hours_attendance", 0), 10) / 10) * 0.04
    
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
    app.run(debug=True)