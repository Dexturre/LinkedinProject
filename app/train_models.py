import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def plot_skills_vs_hire_rate(data_path='app/data/generated_data.csv', output_path='app/static/skills_hire_plot.png'):
    """Create a line plot showing the relationship between skill scores and hire rate"""
    # Load the data
    data = pd.read_csv(data_path)
    
    # Group by skill score ranges and calculate hire rate
    skill_bins = range(0, 101, 5)  # 0-100 in steps of 5
    skill_groups = pd.cut(data['skill_score'], bins=skill_bins, labels=skill_bins[:-1])
    
    hire_rates = []
    skill_centers = []
    
    for i, (skill_range, group) in enumerate(data.groupby(skill_groups)):
        if len(group) > 0:
            hire_rate = group['hire'].mean()
            hire_rates.append(hire_rate)
            skill_centers.append(skill_bins[i] + 2.5)  # Center of the bin
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the line
    plt.plot(skill_centers, hire_rates, marker='o', linewidth=2, markersize=6, 
             color='#2E86AB', label='Hire Rate')
    
    # Add trend line
    z = np.polyfit(skill_centers, hire_rates, 1)
    p = np.poly1d(z)
    plt.plot(skill_centers, p(skill_centers), "--", color='#A23B72', 
             label=f'Trend Line (slope={z[0]:.3f})')
    
    # Customize the plot
    plt.title('Relationship between Skill Score and Hire Rate', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Skill Score', fontsize=14)
    plt.ylabel('Hire Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set axis limits
    plt.xlim(35, 105)
    plt.ylim(-0.1, 1.1)
    
    # Add data points as scatter
    plt.scatter(data['skill_score'], data['hire'], alpha=0.3, s=30, 
                color='#F18F01', label='Individual Candidates')
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_model_comparison(metrics_dict, output_path='app/static/model_comparison.png'):
    """Create a bar chart comparing model performance metrics"""
    models = list(metrics_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])
    
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_distribution_plot(data_path='app/data/generated_data.csv', output_path='app/static/distribution_plot.png'):
    """Create distribution plots for skills and hire decisions"""
    data = pd.read_csv(data_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Skill score distribution
    ax1.hist(data['skill_score'], bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.set_xlabel('Skill Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Skill Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Hire decision distribution
    hire_counts = data['hire'].value_counts()
    colors = ['#F18F01', '#A23B72']
    ax2.pie(hire_counts.values, labels=['Not Hired', 'Hired'], autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax2.set_title('Hire Decision Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# Load dataset
data = pd.read_csv('app/data/generated_data.csv')

# Convert categorical experience levels to numerical values
# Entry Level = 1, Mid Level = 2, Senior Level = 3
experience_mapping = {'Entry Level': 1, 'Mid Level': 2, 'Senior Level': 3}
data['experience_level_numeric'] = data['Experience Level'].map(experience_mapping)

# Split features and target - using all available features including expected_salary_usd
X = data[['experience_level_numeric', 'education_level', 'skill_score', 'expected_salary_usd']]
y = data['hire']  # For classification

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification models
log_reg = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=10)

log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save models
with open('models/logistic_reg.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
with open('models/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)
with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(decision_tree, f)

# Evaluate models
models = {'Logistic Regression': log_reg, 'KNN': knn, 'Decision Tree': decision_tree}
metrics = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1 Score': f1_score(y_test, predictions),
        'Confusion Matrix': confusion_matrix(y_test, predictions).tolist()
    }

# Save metrics to a file
with open('models/metrics.txt', 'w') as f:
    for model_name, model_metrics in metrics.items():
        f.write(f"{model_name}:\n")
        for metric, value in model_metrics.items():
            if metric != 'Confusion Matrix':
                f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write(f"  {metric}: {value}\n")
        f.write("\n")

# Generate visualizations
print("Generating visualizations...")
plot_skills_vs_hire_rate()
plot_model_comparison(metrics)
generate_distribution_plot()
print("All visualizations generated successfully!")

# Print metrics to console
print("\nModel Performance Metrics:")
print("=" * 50)
for model_name, model_metrics in metrics.items():
    print(f"\n{model_name}:")
    for metric, value in model_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
