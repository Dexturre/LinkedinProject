from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load models
try:
    with open('models/logistic_reg.pkl', 'rb') as f:
        logistic_reg = pickle.load(f)
    with open('models/knn.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('models/decision_tree.pkl', 'rb') as f:
        decision_tree_model = pickle.load(f)
except FileNotFoundError:
    print("Models not found. Please run train_models.py first.")
    logistic_reg = None
    knn_model = None
    decision_tree_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([logistic_reg, knn_model, decision_tree_model]):
        return "Models not loaded. Please train models first.", 500
    
    experience = request.form['experience']
    education = request.form['education']
    skill_score = request.form['skill_score']
    
    # Use average salary as default since salary field was removed from frontend
    # Calculate average salary from the dataset
    data = pd.read_csv('app/data/generated_data.csv')
    avg_salary = data['expected_salary_usd'].mean()
    
    # Prepare input data with average salary as default
    input_data = np.array([[int(experience), int(education), int(skill_score), avg_salary]])
    
    # Make predictions using all models
    hire_log_reg = logistic_reg.predict(input_data)[0]
    hire_knn = knn_model.predict(input_data)[0]
    hire_decision_tree = decision_tree_model.predict(input_data)[0]
    
    # Get probability predictions
    log_reg_proba = logistic_reg.predict_proba(input_data)[0][1] if hasattr(logistic_reg, 'predict_proba') else None
    knn_proba = knn_model.predict_proba(input_data)[0][1] if hasattr(knn_model, 'predict_proba') else None
    decision_tree_proba = decision_tree_model.predict_proba(input_data)[0][1] if hasattr(decision_tree_model, 'predict_proba') else None
    
    return render_template('result.html', 
                           hire_log_reg=hire_log_reg,
                           hire_knn=hire_knn,
                           hire_decision_tree=hire_decision_tree,
                           log_reg_proba=log_reg_proba,
                           knn_proba=knn_proba,
                           decision_tree_proba=decision_tree_proba,
                           experience=experience,
                           education=education,
                           skill_score=skill_score,
                           expected_salary=avg_salary)

@app.route('/report')
def report():
    # Load metrics if available
    metrics = {}
    try:
        with open('models/metrics.txt', 'r') as f:
            metrics_content = f.read()
    except FileNotFoundError:
        metrics_content = "Metrics not available. Please run train_models.py first."
    
    # Check if visualization files exist
    skills_plot_exists = os.path.exists('app/static/skills_hire_plot.png')
    model_plot_exists = os.path.exists('app/static/model_comparison.png')
    distribution_plot_exists = os.path.exists('app/static/distribution_plot.png')
    
    return render_template('report.html', 
                          metrics=metrics_content,
                          skills_plot_exists=skills_plot_exists,
                          model_plot_exists=model_plot_exists,
                          distribution_plot_exists=distribution_plot_exists)

@app.route('/data_analysis')
def data_analysis():
    # Load and analyze the data
    try:
        data = pd.read_csv('app/data/generated_data.csv')
        
        # Convert categorical experience levels to numerical values for analysis
        experience_mapping = {'Entry Level': 1, 'Mid Level': 2, 'Senior Level': 3}
        data['experience_level_numeric'] = data['Experience Level'].map(experience_mapping)
        
        # Basic statistics
        total_candidates = len(data)
        hired_count = data['hire'].sum()
        hire_rate = hired_count / total_candidates
        
        # Skill score statistics
        avg_skill = data['skill_score'].mean()
        max_skill = data['skill_score'].max()
        min_skill = data['skill_score'].min()
        
        # Experience statistics (using numeric conversion)
        avg_experience = data['experience_level_numeric'].mean()
        max_experience = data['experience_level_numeric'].max()
        min_experience = data['experience_level_numeric'].min()
        
        # Education statistics
        education_counts = data['education_level'].value_counts()
        
        # Expected salary statistics
        avg_salary = data['expected_salary_usd'].mean()
        max_salary = data['expected_salary_usd'].max()
        min_salary = data['expected_salary_usd'].min()
        
        # Salary suggestions based on experience levels and skills
        salary_suggestions = {
            'entry_level': data[data['Experience Level'] == 'Entry Level']['expected_salary_usd'].mean(),
            'mid_level': data[data['Experience Level'] == 'Mid Level']['expected_salary_usd'].mean(),
            'senior_level': data[data['Experience Level'] == 'Senior Level']['expected_salary_usd'].mean(),
            'high_skill': data[data['skill_score'] >= 80]['expected_salary_usd'].mean(),
            'low_skill': data[data['skill_score'] < 50]['expected_salary_usd'].mean()
        }
        
        return render_template('data_analysis.html',
                             total_candidates=total_candidates,
                             hired_count=hired_count,
                             hire_rate=hire_rate,
                             avg_skill=avg_skill,
                             max_skill=max_skill,
                             min_skill=min_skill,
                             avg_experience=avg_experience,
                             max_experience=max_experience,
                             min_experience=min_experience,
                             education_counts=education_counts,
                             avg_salary=avg_salary,
                             max_salary=max_salary,
                             min_salary=min_salary,
                             salary_suggestions=salary_suggestions)
    except Exception as e:
        return f"Error loading data: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
