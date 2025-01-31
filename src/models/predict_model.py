import joblib
import numpy as np
import pandas as pd
from src.data.preprocessing import DataPreprocessor

class CropPredictor:
    def __init__(self, model_path='models/random_forest_model.joblib'):
        # Load the trained model
        self.model = joblib.load(model_path)
        
        # Initialize preprocessor for scaling
        self.preprocessor = DataPreprocessor()

    def predict(self, soil_data):
        """
        Predict crop for given soil data
        
        :param soil_data: DataFrame or dict with columns/keys 
                          ['Nitrogen', 'Phosphorous', 'Potassium', 'pH']
        :return: Predicted crop(s)
        """
        # Convert input to DataFrame if it's a dictionary
        if isinstance(soil_data, dict):
            soil_data = pd.DataFrame([soil_data])
        
        # Scale the features
        X_scaled = self.preprocessor.feature_scaler.transform(
            soil_data[['Nitrogen', 'Phosphorous', 'Potassium', 'pH']]
        )
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get original crop names
        crop_names = self.preprocessor.get_class_names()
        
        # Map numeric predictions back to crop names
        predicted_crops = [crop_names[pred] for pred in predictions]
        
        return predicted_crops

    def predict_proba(self, soil_data):
        """
        Get prediction probabilities
        """
        # Convert input to DataFrame if it's a dictionary
        if isinstance(soil_data, dict):
            soil_data = pd.DataFrame([soil_data])
        
        # Scale the features
        X_scaled = self.preprocessor.feature_scaler.transform(
            soil_data[['Nitrogen', 'Phosphorous', 'Potassium', 'pH']]
        )
        
        # Get prediction probabilities
        proba = self.model.predict_proba(X_scaled)
        
        # Get original crop names
        crop_names = self.preprocessor.get_class_names()
        
        # Create probability DataFrame
        proba_df = pd.DataFrame(proba, columns=crop_names)
        
        return proba_df