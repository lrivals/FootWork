import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime


# Get the absolute path to the project root directory (3 levels up from the script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from src.Config.Config_Manager import ConfigManager

class StackedFootballPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_model_params):
        """Initialize stacked predictor for binary predictions"""
        self.home_meta_model = RandomForestClassifier(**meta_model_params)
        self.away_meta_model = RandomForestClassifier(**meta_model_params)
        self.binary_models = {}
        self.meta_model_params = meta_model_params
    
    def fit(self, X, y_home, y_away, binary_results):
        """Fit the stacked model using binary classifiers' predictions"""
        self.binary_models = {
            name: results['model'] 
            for name, results in binary_results.items()
        }
        
        meta_features = self._get_meta_features(X)
        
        self.home_meta_model.fit(meta_features, y_home)
        self.away_meta_model.fit(meta_features, y_away)
        
        return self
    
    def _get_meta_features(self, X):
        """Generate meta-features from binary model predictions"""
        return np.column_stack([
            model.predict_proba(X)[:, 1] 
            for model in self.binary_models.values()
        ])
    
    def predict(self, X):
        """Make predictions for both home and away wins"""
        meta_features = self._get_meta_features(X)
        return (
            self.home_meta_model.predict(meta_features),
            self.away_meta_model.predict(meta_features)
        )
    
    def predict_proba(self, X):
        """Get probability predictions for both home and away wins"""
        meta_features = self._get_meta_features(X)
        return (
            self.home_meta_model.predict_proba(meta_features),
            self.away_meta_model.predict_proba(meta_features)
        )

def get_config():
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent / 'Config' / 'configBT_Stacked1.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    return ConfigManager(str(config_path))

def prepare_full_dataset(config):
    paths = config.get_paths()
    exclude_columns = config.get_config_value('excluded_columns', default=[])
    
    df = pd.read_csv(paths['full_dataset'])
    
    df['home_win'] = (df['target_result'] == 'HomeWin').astype(int)
    df['away_win'] = (df['target_result'] == 'AwayWin').astype(int)
    
    X = df.drop(exclude_columns + ['target_result', 'home_win', 'away_win'], axis=1)
    y = df['target_result']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df['home_win'], df['away_win']

def cross_validate_models(X, y, config):
    models = get_models(config)
    n_splits = config.get_config_value('cross_validation', 'n_splits', default=5)
    cv_results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            print(f"\nCross-validating {name}...")
            scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'all_scores': scores
            }
        except Exception as e:
            print(f"Error in cross-validation for {name}: {str(e)}")
            cv_results[name] = {
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'all_scores': np.array([])
            }
    
    return cv_results

def get_models(config):
    """Initialize all models with parameters from config"""
    model_params = config.get_config_value('model_parameters', default={})
    
    return {
        'Random Forest': RandomForestClassifier(**model_params.get('random_forest', {})),
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'SVM': SVC(**model_params.get('svm', {})),
        'Gradient Boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        'XGBoost': XGBClassifier(**model_params.get('xgboost', {})),
        'LightGBM': LGBMClassifier(**model_params.get('lightgbm', {})),
        'CatBoost': CatBoostClassifier(**model_params.get('catboost', {})),
        'Neural Network': MLPClassifier(**model_params.get('neural_network', {})),
        'KNN': KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees': ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }
    
def train_binary_models(X_train, X_test, y_train, y_test, dataset_name, 
                       target_type, config):
    output_dir = config.get_paths()['output_dir']
    models = get_models(config)
    results = {}
    
    cv_results = cross_validate_models(X_train, y_train, config)
    
    for name, model in models.items():
        print(f"\nTraining {name} for {target_type}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'cv_results': cv_results.get(name, {})
            }
            
            save_plots(results[name], name, dataset_name, target_type, output_dir)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def save_plots(results, model_name, dataset_name, target_type, output_dir):
    class_labels = ['Not ' + target_type, target_type]
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name} - {target_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 
        f"confusion_matrix_{dataset_name}_{target_type}_{model_name.replace(' ', '_')}.png"))
    plt.close()
    
    # Cross-validation results if available
    if results['cv_results'].get('all_scores') is not None and len(results['cv_results'].get('all_scores')) > 0:
        plt.figure(figsize=(8, 6))
        plt.boxplot(results['cv_results']['all_scores'])
        plt.title(f'Cross-validation Scores - {model_name} ({target_type})')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(output_dir,
            f"cv_scores_{dataset_name}_{target_type}_{model_name.replace(' ', '_')}.png"))
        plt.close()

def train_stacked_model(config):
    X_full, _, y_home, y_away = prepare_full_dataset(config)
    
    split_params = config.get_config_value('data_split', default={
        'test_size': 0.2,
        'random_state': 42
    })
    
    X_train, X_test, _, _, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X_full, y_home, y_home, y_away, **split_params
    )
    
    binary_results = {}
    
    print("\nProcessing home win predictions...")
    binary_results['home_win'] = train_binary_models(
        X_train, X_test, y_home_train, y_home_test,
        "Binary_Models", "Home_Win", config
    )
    
    print("\nProcessing away win predictions...")
    binary_results['away_win'] = train_binary_models(
        X_train, X_test, y_away_train, y_away_test,
        "Binary_Models", "Away_Win", config
    )
    
    print("\nTraining stacked model...")
    
    try:
        meta_model_params = config.get_config_value('meta_classifier', 'random_forest', default={})
        stacked_model = StackedFootballPredictor(meta_model_params)
        stacked_model.fit(X_train, y_home_train, y_away_train, binary_results['home_win'])
        
        save_stacked_model_results(
            stacked_model, X_test, y_home_test, y_away_test, config
        )
        
        return stacked_model
        
    except Exception as e:
        print(f"Error in stacked model training: {str(e)}")
        raise
def plot_stacked_confusion_matrices(y_home_test, y_home_pred, y_away_test, y_away_pred, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Home win confusion matrix
    home_labels = ['Not Home Win', 'Home Win']
    sns.heatmap(confusion_matrix(y_home_test, y_home_pred), 
               annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=home_labels,
               yticklabels=home_labels)
    ax1.set_title('Home Win Predictions')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Away win confusion matrix
    away_labels = ['Not Away Win', 'Away Win']
    sns.heatmap(confusion_matrix(y_away_test, y_away_pred),
               annot=True, fmt='d', cmap='Blues', ax=ax2,
               xticklabels=away_labels,
               yticklabels=away_labels)
    ax2.set_title('Away Win Predictions')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_model_binary_confusion_matrices.png"))
    plt.close()

def save_stacked_model_results(model, X_test, y_home_test, y_away_test, config):
    output_dir = config.get_paths()['output_dir']
    y_home_pred, y_away_pred = model.predict(X_test)
    y_home_prob, y_away_prob = model.predict_proba(X_test)
    
    results_file = os.path.join(output_dir, "stacked_model_binary_results.txt")
    with open(results_file, 'w') as f:
        f.write("Stacked Model Binary Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Home Win Results
        f.write("\nHome Win Predictions:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {accuracy_score(y_home_test, y_home_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_home_test, y_home_pred, 
                                   target_names=['Not Home Win', 'Home Win']))
        
        # Away Win Results
        f.write("\nAway Win Predictions:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {accuracy_score(y_away_test, y_away_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_away_test, y_away_pred,
                                   target_names=['Not Away Win', 'Away Win']))

def main():
    try:
        config = get_config()
        stacked_model = train_stacked_model(config)
        print("\nStacked model training completed. Results saved in output directory.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()