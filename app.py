from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import numpy as np
import os
import re
from werkzeug.utils import secure_filename

# For CV parsing
import PyPDF2
import docx2txt
import tempfile
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

os.makedirs('model', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

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

# Define common technology keywords for skill detection
tech_keywords = [
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "ruby", "go", "php", "swift", "kotlin", 
    "typescript", "rust", "scala", "perl", "r", "matlab", "bash", "shell", "sql", "html", "css",
    
    # Frameworks & Libraries
    "django", "flask", "spring", "react", "angular", "vue", "node.js", "express", 
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "bootstrap", "jquery",
    "laravel", "symfony", "rails", ".net", "asp.net", "xamarin", "flutter", "electron",
    
    # Databases
    "mysql", "postgresql", "mongodb", "oracle", "sql server", "sqlite", "redis", "cassandra",
    "dynamodb", "firebase", "mariadb", "neo4j", "elasticsearch", "couchdb", "influxdb",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins", "terraform",
    "ansible", "chef", "puppet", "circleci", "travis", "github actions", "gitlab ci",
    "prometheus", "grafana", "elk stack", "cloudformation", "vagrant", "heroku", "vercel",
    
    # Other Technologies
    "rest api", "graphql", "grpc", "oauth", "jwt", "microservices", "serverless",
    "pwa", "webrtc", "websocket", "ajax", "json", "xml", "yaml", "markdown",
    "git", "svn", "mercurial", "linux", "unix", "windows", "macos", "ios", "android"
]

# Define common certifications
common_certifications = [
    "aws certified", "azure certified", "google cloud certified", "comptia", "cisco ccna", 
    "cisco ccnp", "cisco ccie", "pmp", "prince2", "scrum", "safe", "itil", "six sigma",
    "ceh", "cissp", "cisa", "cism", "oscp", "security+", "network+", "a+", "linux+",
    "oracle certified", "ibm certified", "vmware certified", "salesforce certified",
    "red hat certified", "microsoft certified", "mcsa", "mcse", "mcts", "mcitp",
    "togaf", "cka", "ckad", "cna", "cnsp", "professional scrum", "pmi", "agile",
    "hadoop", "spark", "tensorflow", "data science", "machine learning", "ai certification",
    "java certification", "python certification", "javascript certification", "frontend"
]

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

@app.route("/analyze-cv", methods=["POST"])
def analyze_cv():
    """Analyze CV and extract relevant information"""
    try:
        print("Analyzing CV...")
        if 'cv' not in request.files:
            print("No file part in request")
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['cv']
        
        if file.filename == '':
            print("No selected file")
            return jsonify({"error": "No selected file"}), 400
            
        # Check if the file has a valid extension
        allowed_extensions = {'pdf', 'doc', 'docx', 'txt', 'rtf'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            print("Invalid file type")
            return jsonify({"error": "Invalid file type"}), 400
            
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name
        
        # Extract text from the CV based on file type
        text = extract_text_from_cv(temp_filename, file_extension)
        
        # Remove the temporary file
        os.unlink(temp_filename)
        
        if not text:
            print("Could not extract text from the file")
            return jsonify({"error": "Could not extract text from the file"}), 400
            
        # Analyze the CV text to extract information
        extracted_data = extract_information_from_cv(text)
        
        print("Extracted data:", extracted_data)
        return jsonify(extracted_data)
    except Exception as e:
        print(f"Error in CV analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_text_from_cv(file_path, file_extension):
    """Extract text from CV based on file type"""
    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension in ['doc', 'docx']:
            text = extract_text_from_docx(file_path)
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_extension == 'rtf':
            # Simple RTF to text conversion
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
                # Remove RTF formatting
                text = re.sub(r'\\[a-z0-9]+', ' ', rtf_content)
                text = re.sub(r'\{|\}|\\\n|\\\r|\\_|\\\-|\\~|\\\'|\\"', ' ', text)
        else:
            text = ""
            
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text += page.extract_text() + "\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_information_from_cv(text):
    """Extract relevant information from CV text"""
    # Normalize text: convert to lowercase and remove extra whitespace
    text = ' '.join(text.lower().split())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Define result structure
    result = {
        "years_of_experience": estimate_years_of_experience(text),
        "technical_skills": count_technical_skills(text, tokens),
        "certificates": count_certificates(text),
        "personal_projects": estimate_personal_projects(text),
        "academic_achievements": estimate_academic_achievements(text),
        "extra_academic_achievements": estimate_extra_academic_achievements(text),
        "networking_referees": estimate_professional_references(text)
    }
    
    return result

def estimate_years_of_experience(text):
    """Estimate years of experience from CV text"""
    # Look for patterns like "X years of experience" or "experienced for X years"
    experience_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?',
        r'(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?\s+(?:in\s+)(?:the\s+)?(?:field|industry)',
        r'worked\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?'
    ]
    
    years = []
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                years.append(float(match.group(1)))
            except (IndexError, ValueError):
                continue
    
    # If explicit years of experience found, use the largest value
    if years:
        return max(years)
    
    # If not found, estimate from work history
    # Look for date ranges in work experience section
    date_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\s*(?:-|to|–|—)\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|present|current|now)'
    date_matches = re.findall(date_pattern, text, re.IGNORECASE)
    
    if date_matches:
        # Estimate 1 year of experience for each date range found
        years_estimate = min(len(date_matches), 15)  # Cap at 15 years
        return years_estimate
    
    # If no dates found, check for job titles as a proxy for experience
    job_titles = [
        'senior', 'lead', 'principal', 'staff', 'manager', 'director', 'head', 
        'architect', 'cto', 'vp', 'chief'
    ]
    
    for title in job_titles:
        if re.search(r'\b' + title + r'\b', text, re.IGNORECASE):
            # Senior positions suggest more experience
            if title in ['senior', 'lead']:
                return 5
            elif title in ['principal', 'staff', 'manager']:
                return 7
            elif title in ['director', 'head', 'architect']:
                return 10
            elif title in ['cto', 'vp', 'chief']:
                return 12
    
    # Default: assume entry-level position with 1 year of experience
    return 1

def count_technical_skills(text, tokens):
    """Count the number of technical skills mentioned in the CV"""
    skill_count = 0
    
    # Check for presence of each keyword in the text
    for keyword in tech_keywords:
        # Use word boundaries to match whole words
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            skill_count += 1
    
    # Cap at 10 skills
    return min(skill_count, 10)

def count_certificates(text):
    """Count the number of certificates mentioned in the CV"""
    cert_count = 0
    
    # Look for certification keywords
    for cert in common_certifications:
        if re.search(r'\b' + re.escape(cert) + r'\b', text, re.IGNORECASE):
            cert_count += 1
    
    # Look for words like "certified", "certificate", "certification"
    cert_mentions = len(re.findall(r'\b(?:certified|certificate|certification)\b', text, re.IGNORECASE))
    
    # Use the larger of the two counts, capped at 5
    return min(max(cert_count, cert_mentions), 5)

def estimate_personal_projects(text):
    """Estimate the number of personal projects from CV text"""
    # Look for sections that might contain projects
    project_section = False
    project_keywords = ['project', 'portfolio', 'github', 'gitlab', 'bitbucket', 'repository']
    
    for keyword in project_keywords:
        if re.search(r'\b' + re.escape(keyword) + r's?\b', text, re.IGNORECASE):
            project_section = True
            break
    
    if not project_section:
        return 0
    
    # Count project indicators
    # Look for bullet points or numbered items in project sections
    bullet_pattern = r'(?:•|\*|-|–|—|\d+\.)\s+\w+'
    bullets = re.findall(bullet_pattern, text)
    
    # Look for GitHub/repository links
    repo_links = re.findall(r'github\.com/[^\s]+', text)
    
    # Estimate project count from bullet points and links
    project_estimate = len(repo_links) + min(len(bullets) // 3, 5)  # Assume ~3 bullets per project
    
    # Cap at 8 projects
    return min(project_estimate, 8)

def estimate_academic_achievements(text):
    """Estimate academic achievements score from CV text"""
    base_score = 70  # Start with a default score
    
    # Look for education level
    education_levels = {
        'phd': 15,
        'doctorate': 15,
        'master': 10,
        'mba': 10,
        'bachelor': 5,
        'undergraduate': 5,
        'diploma': 3,
        'certificate': 2
    }
    
    for level, points in education_levels.items():
        if re.search(r'\b' + re.escape(level) + r'\w*\b', text, re.IGNORECASE):
            base_score += points
            break
    
    # Look for GPA or grades
    gpa_match = re.search(r'gpa\s*(?:of|:)?\s*(\d+\.\d+)', text, re.IGNORECASE)
    if gpa_match:
        try:
            gpa = float(gpa_match.group(1))
            # Assume GPA is on a 4.0 scale
            if gpa > 3.7:
                base_score += 10
            elif gpa > 3.5:
                base_score += 7
            elif gpa > 3.0:
                base_score += 5
        except (IndexError, ValueError):
            pass
    
    # Look for honors or distinctions
    honors = [
        'summa cum laude', 'magna cum laude', 'cum laude', 'honors', 'distinction',
        'dean\'s list', 'scholarship', 'award', 'fellowship', 'valedictorian'
    ]
    
    for honor in honors:
        if re.search(r'\b' + re.escape(honor) + r'\b', text, re.IGNORECASE):
            base_score += 5
            break
    
    # Cap the score at 100
    return min(base_score, 100)

def estimate_extra_academic_achievements(text):
    """Estimate number of extra-academic achievements"""
    # Look for common extra-academic activities
    activities = [
        'volunteer', 'volunteering', 'community service',
        'leadership', 'club president', 'organization founder',
        'competition', 'hackathon', 'contest', 'challenge',
        'award', 'recognition', 'honor',
        'publication', 'research paper', 'journal article',
        'presentation', 'conference', 'speaking engagement',
        'mentor', 'tutor', 'teaching assistant'
    ]
    
    achievement_count = 0
    for activity in activities:
        if re.search(r'\b' + re.escape(activity) + r'\b', text, re.IGNORECASE):
            achievement_count += 1
    
    # Cap at 5 achievements
    return min(achievement_count, 5)

def estimate_professional_references(text):
    """Estimate number of professional references"""
    # Look for reference indicators
    reference_keywords = [
        'reference', 'referral', 'recommendation', 'referee', 'testimonial'
    ]
    
    for keyword in reference_keywords:
        if re.search(r'\b' + re.escape(keyword) + r's?\b', text, re.IGNORECASE):
            # Check for common reference patterns
            # Count email patterns in the vicinity of reference keywords
            context_window = 500  # Characters
            keyword_match = re.search(r'\b' + re.escape(keyword) + r's?\b', text, re.IGNORECASE)
            
            if keyword_match:
                start_pos = max(0, keyword_match.start() - context_window)
                end_pos = min(len(text), keyword_match.end() + context_window)
                context = text[start_pos:end_pos]
                
                # Count emails in context
                emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', context)
                
                # Count phone numbers in context
                phones = re.findall(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b', context)
                
                # Return the count of emails and phones, capped at 5
                return min(len(emails) + len(phones), 5)
    
    # If we found a "References available upon request" statement
    if re.search(r'references\s+(?:are\s+)?available\s+(?:upon|on)\s+request', text, re.IGNORECASE):
        return 2  # Assume they have at least a couple of references
    
    # Default: assume 1 reference
    return 1

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