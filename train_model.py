import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train_student_success_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Feature engineering
    features = [
        "previous_gpa", "assignment_completion", "study_hours_weekly",
        "submission_timeliness", "debugging_time", "forum_participation",
        "office_hours_attendance"
    ]
    
    X = df[features]
    y = df["passed_course"]  # Target variable
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=5, n_jobs=-1, verbose=2, scoring='f1')
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test_scaled)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    print(f"Model performance metrics: {metrics}")
    
    # Save model and scaler
    with open("model/student_success_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return best_model, metrics

if __name__ == "__main__":
    train_student_success_model("data/student_performance.csv")