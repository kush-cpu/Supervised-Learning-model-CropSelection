import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

class DataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)

    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic agricultural dataset
        """
        np.random.seed(42)
        
        # Generate realistic features
        nitrogen = np.concatenate([
            np.random.normal(30, 10, n_samples // 3),
            np.random.normal(70, 15, n_samples // 3),
            np.random.normal(110, 20, n_samples // 3)
        ])
        
        phosphorous = np.concatenate([
            np.random.normal(20, 5, n_samples // 3),
            np.random.normal(60, 10, n_samples // 3),
            np.random.normal(100, 15, n_samples // 3)
        ])
        
        potassium = np.concatenate([
            np.random.normal(50, 15, n_samples // 3),
            np.random.normal(100, 20, n_samples // 3),
            np.random.normal(150, 25, n_samples // 3)
        ])
        
        ph = np.concatenate([
            np.random.normal(5.5, 0.5, n_samples // 3),
            np.random.normal(6.5, 0.3, n_samples // 3),
            np.random.normal(7.5, 0.5, n_samples // 3)
        ])
        
        # Clip values to realistic ranges
        nitrogen = np.clip(nitrogen, 0, 140)
        phosphorous = np.clip(phosphorous, 5, 145)
        potassium = np.clip(potassium, 5, 205)
        ph = np.clip(ph, 3, 10)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Nitrogen': nitrogen,
            'Phosphorous': phosphorous,
            'Potassium': potassium,
            'pH': ph
        })
        
        # Crop selection logic
        crops = []
        for i in range(len(data)):
            if ph[i] < 5.5:
                crop = np.random.choice(['blueberry', 'potato'], p=[0.7, 0.3])
            elif ph[i] < 6.5:
                crop = np.random.choice(['carrot', 'corn'], p=[0.5, 0.5])
            elif ph[i] < 7.5:
                crop = np.random.choice(['tomato', 'soybean'], p=[0.5, 0.5])
            else:
                crop = np.random.choice(['barley', 'spinach'], p=[0.6, 0.4])
            
            crops.append(crop)
        
        data['Crop'] = crops
        return data

    def save_data(self, data: pd.DataFrame, filename: str = 'synthetic_crop_data.csv'):
        """
        Save processed data to CSV
        """
        output_path = os.path.join(self.processed_data_path, filename)
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")