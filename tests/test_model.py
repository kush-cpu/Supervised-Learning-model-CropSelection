import pytest
import numpy as np
from src.data.make_dataset import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.predict_model import CropPredictor

def test_model_training():
    """
    Test model training process
    """
    # Load synthetic data
    data_loader = DataLoader()
    synthetic_data = data_loader.generate_synthetic_data()
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(synthetic_data)
    crop_classes = preprocessor.get_class_names()
    
    # Train models
    trainer = ModelTrainer()
    models = trainer.train_models(X_train, y_train)
    
    # Check models are trained
    assert len(models) > 0
    for name, model in models.items():
        assert hasattr(model, 'predict')

def test_crop_prediction():
    """
    Test crop prediction functionality
    """
    # Sample soil data
    sample_soil = {
        'Nitrogen': 75,
        'Phosphorous': 50,
        'Potassium': 100,
        'pH': 6.5
    }
    
    # Create predictor
    predictor = CropPredictor()
    
    # Test prediction
    predicted_crops = predictor.predict(sample_soil)
    assert len(predicted_crops) > 0
    
    # Test probability prediction
    crop_probabilities = predictor.predict_proba(sample_soil)
    assert not crop_probabilities.empty
    assert crop_probabilities.sum(axis=1).all() == pytest.approx(1.0)