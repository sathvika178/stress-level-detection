import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class StressLevelModel:
    """
    Machine learning model for stress level detection based on environmental and physiological data.
    """
    def __init__(self):
        """Initialize the stress level detection model"""
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.scaler = StandardScaler()
        self.features = ['Humidity', 'Temperature', 'Step count']
        self.target = 'Stress Level'
        self.is_trained = False
        self.feature_importances = {}
        
    def preprocess_data(self, data):
        """
        Preprocess the input data for the model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to preprocess
            
        Returns:
        --------
        X : numpy.ndarray
            Preprocessed features
        y : numpy.ndarray
            Target values (if available)
        """
        # Check if target column exists in data
        has_target = self.target in data.columns
        
        # Separate features and target
        X = data[self.features].values
        y = data[self.target].values if has_target else None
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, data):
        """
        Train the model on the provided data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The training data with features and target
            
        Returns:
        --------
        metrics : dict
            Training performance metrics
        """
        # Preprocess data
        X, y = self.preprocess_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store feature importances
        self.feature_importances = dict(zip(self.features, self.model.feature_importances_))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to make predictions on
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted stress levels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Extract features
        X = data[self.features].values
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_feature_importances(self):
        """
        Get the feature importances of the trained model
        
        Returns:
        --------
        feature_importances : dict
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        return self.feature_importances
    
    def save_model(self, path="model_files"):
        """
        Save the trained model to disk
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, os.path.join(path, "stress_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        
    def load_model(self, path="model_files"):
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        # Load model and scaler
        self.model = joblib.load(os.path.join(path, "stress_model.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.is_trained = True
