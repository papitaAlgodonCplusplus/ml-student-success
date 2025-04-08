# Student Success Prediction System

![Student Success Prediction Banner](https://via.placeholder.com/800x200?text=Student+Success+Prediction+System)

## About the Project

The Student Success Prediction System is an AI-powered web application designed to help students assess their likelihood of success in programming courses. By analyzing various academic and behavioral factors, the system provides personalized insights and actionable recommendations to improve learning outcomes.

## How It Works

The system uses a machine learning model to analyze key predictors of academic success including:

- Previous GPA
- Assignment completion rate
- Weekly study hours
- Submission timeliness
- Time spent debugging
- Forum participation
- Office hours attendance

Based on these factors, the application predicts the probability of success in a programming course and offers tailored recommendations to improve performance.

## The Model

Our prediction engine uses a **Random Forest Classifier** trained on anonymized student performance data. This model was chosen for its:

- High accuracy (~94% on test data)
- Ability to handle different types of input features
- Resistance to overfitting
- Interpretability of feature importance

The model provides not just predictions, but also insights into which factors most significantly impact success, allowing for targeted improvement strategies.

## How to Use the Application

1. Visit [https://student-success-predictor.herokuapp.com](https://student-success-predictor.herokuapp.com)
2. Enter your academic information in the input form:
   - Previous GPA (0-4.0)
   - Assignment completion rate (%)
   - Weekly study hours
   - Submission timeliness (%)
   - Average debugging time (hours/week)
   - Forum participation score (0-100)
   - Number of office hours attended
3. Click "Predict Success Probability"
4. Review your results, which include:
   - Overall success probability
   - Risk level assessment
   - Visualization of key factors affecting your outcome
   - Personalized recommendations for improvement

## Example Use Cases

### For Students
- Identify areas where additional effort would most improve your chances of success
- Create a data-driven study plan based on personalized recommendations
- Track progress over time by updating inputs as behaviors change

### For Teaching Assistants
- Help students understand which study behaviors to prioritize
- Provide structured guidance during office hours
- Identify common patterns among struggling students

### For Instructors
- Gain insights into class-wide success factors
- Develop targeted interventions for at-risk students
- Structure course components to address common weak points

## Privacy and Data Use

This application does not store any personal data. All predictions are generated in real-time based on your inputs, and this information is not saved after you close the application. No cookies or tracking mechanisms are used.

## Technical Details

- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Chart.js
- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Deployment**: Heroku

## Feedback and Improvements

We're continuously working to improve the prediction accuracy and user experience. If you have suggestions or feedback, please contact us at [contact@student-success-predictor.edu](mailto:contact@student-success-predictor.edu).

## Local Development

For developers interested in running the application locally:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/student-success-predictor.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at http://localhost:5000

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Disclaimer: This tool provides estimates based on historical data and should be used as one of many resources to guide academic decisions, not as the sole determinant of study strategies.*