from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    
    def train(self, X_train, y_train):
        """Train the decision tree model"""
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
