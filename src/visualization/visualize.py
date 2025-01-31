import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, data):
        """
        Initialize visualizer with dataset
        
        :param data: pandas DataFrame with crop and soil feature data
        """
        self.data = data

    def plot_feature_distributions(self, features=None):
        """
        Plot distribution of soil features
        
        :param features: List of features to plot, default is all numeric columns
        :return: matplotlib figure
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != 'Crop']
        
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(features, 1):
            plt.subplot(1, len(features), i)
            sns.histplot(data=self.data, x=feature, hue='Crop', multiple='stack', kde=True)
            plt.title(f'{feature} Distribution by Crop')
        
        plt.tight_layout()
        return plt

    def plot_correlation_heatmap(self, features=None):
        """
        Create correlation heatmap for soil features
        
        :param features: List of features to include in correlation
        :return: matplotlib figure
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != 'Crop']
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Soil Feature Correlation Heatmap')
        return plt

    def plot_crop_distribution(self):
        """
        Plot distribution of crops
        
        :return: matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        crop_counts = self.data['Crop'].value_counts()
        sns.barplot(x=crop_counts.index, y=crop_counts.values)
        plt.title('Crop Distribution')
        plt.xlabel('Crop')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        return plt

    def plot_boxplot_by_crop(self, features=None):
        """
        Create boxplots of features for each crop
        
        :param features: List of features to plot
        :return: matplotlib figure
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != 'Crop']
        
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(features, 1):
            plt.subplot(1, len(features), i)
            sns.boxplot(x='Crop', y=feature, data=self.data)
            plt.title(f'{feature} by Crop')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt
