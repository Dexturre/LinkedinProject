from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
    
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_model(self):
        """Return the trained model"""
        return self.model
