from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def train(self, X_train, y_train):
        """Train the KNN model"""
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
