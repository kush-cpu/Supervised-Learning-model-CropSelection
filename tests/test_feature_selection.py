import pytest
import numpy as np
import pandas as pd
from src.features.feature_selection import FeatureSelector
from src.data.make_dataset import DataLoader
from src.data.preprocessing import DataPreprocessor

def test_feature_selection():
    """
    Test feature selection methods
    """
    # Load synthetic data
    data_loader = DataLoader()
    synthetic_data = data_loader.generate_synthetic_data()
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X = synthetic_data[['Nitrogen', 'Phosphorous', 'Potassium', 'pH']]
    y = preprocessor.label_encoder.transform(synthetic_data['Crop'])
    
    # Perform feature selection
    selector = FeatureSelector()
    feature_scores = selector.select_features(X, y)
    
    # Check results
    assert len(feature_scores) > 0
    for method, scores in feature_scores.items():
        assert not scores.empty
        assert 'Feature' in scores.columns
        assert 'Score' in scores.columns
