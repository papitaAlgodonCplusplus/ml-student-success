from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load the trained model
with open("model/student_success_model.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from request
    data = request.json
    
    # Create dataframe from input
    input_df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_probability = model.predict_proba(input_df)[0]
    
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
    
    response = {
        "prediction": float(prediction),
        "success_probability": float(prediction_probability[1]),
        "risk_level": get_risk_level(prediction_probability[1]),
        "recommendations": recommendations,
        "explanation": explanation
    }
    
    return jsonify(response)

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
    
    # Add more recommendation logic based on other features
    
    return recommendations

if __name__ == "__main__":
    app.run(debug=True)