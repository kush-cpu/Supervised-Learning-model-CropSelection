import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning
        """
        # Separate features and target
        X = data[['Nitrogen', 'Phosphorous', 'Potassium', 'pH']]
        y = self.label_encoder.fit_transform(data['Crop'])
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test

    def get_class_names(self) -> np.ndarray:
        """
        Return the original crop names
        """
        return self.label_encoder.classes_