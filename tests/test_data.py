import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import DataLoader
from src.data.preprocessing import DataPreprocessor

def test_data_generation():
    """
    Test synthetic data generation
    """
    data_loader = DataLoader()
    synthetic_data = data_loader.generate_synthetic_data()
    
    # Check data properties
    assert isinstance(synthetic_data, pd.DataFrame)
    assert len(synthetic_data) > 0
    assert 'Crop' in synthetic_data.columns
    
    # Check feature ranges
    features = ['Nitrogen', 'Phosphorous', 'Potassium', 'pH']
    for feature in features:
        assert synthetic_data[feature].min() >= 0
        assert synthetic_data[feature].max() <= 200  # Adjust based on your ranges

def test_data_preprocessing():
    """
    Test data preprocessing
    """
    data_loader = DataLoader()
    synthetic_data = data_loader.generate_synthetic_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(synthetic_data)
    
    # Check shapes
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 4  # Number of features
    
    # Check target encoding
    crop_classes = preprocessor.get_class_names()
    assert len(crop_classes) > 0