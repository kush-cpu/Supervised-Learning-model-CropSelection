import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()

    def select_features(self, X, y, methods=['mutual_info', 'f_classif', 'chi2']):
        """
        Apply multiple feature selection methods
        """
        results = {}
        
        # Prepare data for chi-square (needs non-negative scaled data)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.abs(X_scaled)

        # Feature selection methods
        method_funcs = {
            'mutual_info': mutual_info_classif,
            'f_classif': f_classif,
            'chi2': chi2
        }

        for method_name in methods:
            method = method_funcs[method_name]
            
            if method_name == 'chi2':
                scores = method(X_scaled, y)[0]
            elif method_name == 'f_classif':
                scores = method(X, y)[0]
            else:
                scores = method(X, y)

            results[method_name] = pd.DataFrame({
                'Feature': X.columns,
                'Score': scores
            }).sort_values('Score', ascending=False)

        return results

    def plot_feature_importance(self, feature_scores):
        """
        Create comparison plot of feature importance across methods
        """
        plt.figure(figsize=(15, 5))
        
        for i, (method_name, scores_df) in enumerate(feature_scores.items(), 1):
            plt.subplot(1, len(feature_scores), i)
            sns.barplot(x='Feature', y='Score', data=scores_df)
            plt.title(f'Feature Importance ({method_name})')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plt

    def recommend_best_feature(self, feature_scores):
        """
        Recommend the best feature based on average ranking
        """
        # Combine rankings from different methods
        combined_ranking = {}
        for method, scores_df in feature_scores.items():
            for rank, row in scores_df.iterrows():
                feature = row['Feature']
                if feature not in combined_ranking:
                    combined_ranking[feature] = []
                combined_ranking[feature].append(rank)

        # Calculate average rank for each feature
        average_ranks = {
            feature: np.mean(ranks) 
            for feature, ranks in combined_ranking.items()
        }

        # Return feature with lowest average rank (best feature)
        best_feature = min(average_ranks, key=average_ranks.get)
        return best_feature, average_ranks