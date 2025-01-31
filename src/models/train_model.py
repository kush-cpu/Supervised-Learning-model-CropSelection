import numpy as np
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.random_state = self.config['model']['random_seed']

    def train_models(self, X_train, y_train):
        """
        Train multiple models
        """
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Logistic Regression': LogisticRegression(multi_class='multinomial', random_state=self.random_state),
            'Support Vector Machine': SVC(random_state=self.random_state)
        }

        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model

        return trained_models

    def evaluate_models(self, models, X_test, y_test, class_names):
        """
        Evaluate trained models
        """
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(
                    y_test, y_pred, target_names=class_names
                )
            }
        return results
    def save_models(self, models, path='models/'):
        """
        Save trained models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in models.items():
            filename = os.path.join(path, f'{name.lower().replace(" ", "_")}_model.joblib')
            joblib.dump(model, filename)
        
        print("Models saved successfully.")