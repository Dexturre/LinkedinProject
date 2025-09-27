# AI Smart Job Candidate Screening System

An intelligent web application that uses machine learning to predict candidate hiring decisions based on experience, education, and skill scores.

## Features

- **Three ML Models**: Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree
- **Interactive Web Interface**: Flask-based web application with modern UI
- **Data Visualization**: Matplotlib integration for skills vs hire rate analysis
- **Model Comparison**: Comprehensive performance metrics and visualizations
- **Data Analysis**: Statistical insights and distribution analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI_Smart_Job_Candidate_Screening
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Train the models** (first time setup):
```bash
python app/train_models.py
```

2. **Start the web application**:
```bash
python app/app.py
```

3. **Access the application**:
   - Open your browser and go to `http://localhost:5000`
   - Use the form to input candidate data and get predictions
   - View model reports and data analysis

## Project Structure

```
app/
├── Models/                 # Custom ML model implementations
│   ├── logistic_reg.py    # Logistic Regression model
│   ├── KNN.py            # K-Nearest Neighbors model
│   └── decision_tree.py  # Decision Tree model
├── data/                  # Dataset
│   └── candidatess_100.csv
├── static/               # Static files (CSS, images)
│   ├── style.css
│   └── *.png            # Generated visualizations
├── templates/            # HTML templates
│   ├── index.html       # Main input form
│   ├── result.html      # Prediction results
│   ├── report.html      # Model evaluation report
│   └── data_analysis.html # Data statistics
├── app.py               # Flask application
├── train_models.py      # Model training script
└── visualization.py     # Data visualization functions
```

## Machine Learning Models

The system uses three different machine learning algorithms:

1. **Logistic Regression**: Linear model for binary classification
2. **K-Nearest Neighbors**: Instance-based learning algorithm
3. **Decision Tree**: Non-linear tree-based classifier

## Data Features

- **Experience**: Years of professional experience (numeric)
- **Education**: Education level (1: Bachelor, 2: Master, 3: PhD)
- **SkillScore**: Technical skill assessment score (0-100)
- **Hire (0/1)**: Target variable (0: Not hired, 1: Hired)

## Visualization Features

- **Skills vs Hire Rate**: Line plot showing correlation between skill scores and hiring decisions
- **Model Performance Comparison**: Bar charts comparing accuracy, precision, recall, and F1 scores
- **Data Distribution**: Histograms and pie charts showing dataset characteristics

## API Endpoints

- `/` - Main prediction form
- `/predict` - POST endpoint for making predictions
- `/report` - Model evaluation report with visualizations
- `/data_analysis` - Statistical analysis of the dataset

## Dependencies

- Flask (Web framework)
- pandas (Data manipulation)
- numpy (Numerical computing)
- scikit-learn (Machine learning)
- matplotlib (Data visualization)
- seaborn (Statistical data visualization)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with Python and Flask
- Uses scikit-learn for machine learning
- Matplotlib for data visualization
- Modern web design principles
