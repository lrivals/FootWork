import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path to import the binary prediction module
sys.path.append(str(Path(__file__).parent.parent.parent))
from Config.Config_Manager import ConfigManager

class StackedFootballPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_model_params):
        """Initialize stacked predictor for binary predictions"""
        # Create separate meta-models for home and away predictions
        self.home_meta_model = RandomForestClassifier(**meta_model_params)
        self.away_meta_model = RandomForestClassifier(**meta_model_params)
        self.binary_models = {}
    
    def fit(self, X, y_home, y_away, binary_results):
        """
        Fit the stacked model using binary classifiers' predictions
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_home : array-like
            Binary target for home wins
        y_away : array-like
            Binary target for away wins
        binary_results : dict
            Dictionary containing trained binary models
        """
        # Store binary models
        self.binary_models = {
            name: results['model'] 
            for name, results in binary_results.items()
        }
        
        # Generate meta-features
        meta_features = self._get_meta_features(X)
        
        # Fit meta-models for both targets
        self.home_meta_model.fit(meta_features, y_home)
        self.away_meta_model.fit(meta_features, y_away)
        
        return self
    
    def _get_meta_features(self, X):
        """Generate meta-features from binary model predictions"""
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.binary_models.values()
        ])
        return meta_features
    
    def predict(self, X):
        """
        Make predictions for both home and away wins
        Returns tuple of (home_predictions, away_predictions)
        """
        meta_features = self._get_meta_features(X)
        home_predictions = self.home_meta_model.predict(meta_features)
        away_predictions = self.away_meta_model.predict(meta_features)
        return home_predictions, away_predictions
    
    def predict_proba(self, X):
        """
        Get probability predictions for both home and away wins
        Returns tuple of (home_probabilities, away_probabilities)
        """
        meta_features = self._get_meta_features(X)
        home_probabilities = self.home_meta_model.predict_proba(meta_features)
        away_probabilities = self.away_meta_model.predict_proba(meta_features)
        return home_probabilities, away_probabilities

def get_config():
    """Get configuration with validation"""
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent / 'Config' / 'configBT_Stacked1.yaml'
    
    print(f"\nAttempting to load config from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    # Load and print config contents for debugging
    with open(config_path, 'r') as f:
        config_contents = yaml.safe_load(f)
        print("\nLoaded configuration contents:")
        print(yaml.dump(config_contents, default_flow_style=False))
    
    return ConfigManager(str(config_path))

def prepare_full_dataset(input_path, exclude_columns):
    """
    Prepare the full dataset keeping all samples and creating binary targets
    """
    df = pd.read_csv(input_path)
    
    # Create binary targets for both win types
    df['home_win'] = (df['target_result'] == 'HomeWin').astype(int)
    df['away_win'] = (df['target_result'] == 'AwayWin').astype(int)
    
    # Create feature matrix X and original target y
    X = df.drop(exclude_columns + ['target_result', 'home_win', 'away_win'], axis=1)
    y = df['target_result']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df['home_win'], df['away_win']

def cross_validate_models(X, y, model_params, n_splits=5):
    """Perform cross-validation for each model type"""
    models = {
        'Random Forest': RandomForestClassifier(**model_params['random_forest']),
        'Logistic Regression': LogisticRegression(**model_params['logistic_regression']),
        'SVM': SVC(**model_params['svm'])
    }
    
    cv_results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        cv_results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'all_scores': scores
        }
    
    return cv_results

def train_binary_models(X_train, X_test, y_train, y_test, dataset_name, 
                       target_type, output_dir, model_params):
    """Train and evaluate binary models"""
    models = {
        'Random Forest': RandomForestClassifier(**model_params['random_forest']),
        'Logistic Regression': LogisticRegression(**model_params['logistic_regression']),
        'SVM': SVC(**model_params['svm'])
    }
    
    results = {}
    class_labels = ['Not ' + target_type, target_type]
    
    # Perform cross-validation
    cv_results = cross_validate_models(X_train, y_train, model_params)
    
    for name, model in models.items():
        print(f"\nTraining {name} for {target_type}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'cv_results': cv_results[name]
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {name} ({dataset_name} - {target_type})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 
            f"confusion_matrix_{dataset_name}_{target_type}_{name.replace(' ', '_')}.png"))
        plt.close()
        
        # Plot cross-validation results
        plt.figure(figsize=(8, 6))
        plt.boxplot(cv_results[name]['all_scores'])
        plt.title(f'Cross-validation Scores - {name} ({target_type})')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(output_dir,
            f"cv_scores_{dataset_name}_{target_type}_{name.replace(' ', '_')}.png"))
        plt.close()
    
    return results

def train_stacked_model(config):
    """Train and evaluate the stacked model for binary predictions"""
    paths = config.get_paths()
    exclude_columns = config.get_excluded_columns()
    split_params = config.get_split_params()
    model_params = config.get_model_params()
    meta_model_params = config.config['meta_classifier']['random_forest']
    
    # Prepare the full dataset
    print("\nPreparing full dataset...")
    X_full, _, y_home, y_away = prepare_full_dataset(paths['full_dataset'], exclude_columns)
    
    # Split the full dataset
    X_train, X_test, _, _, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X_full, y_home, y_home, y_away, **split_params
    )
    
    # Train binary models and collect results
    binary_results = {}
    
    # Train home win classifier
    print("\nProcessing home win predictions...")
    binary_results['home_win'] = train_binary_models(
        X_train, X_test, y_home_train, y_home_test,
        "Binary_Models", "Home_Win", 
        paths['output_dir'], model_params
    )
    
    # Train away win classifier
    print("\nProcessing away win predictions...")
    binary_results['away_win'] = train_binary_models(
        X_train, X_test, y_away_train, y_away_test,
        "Binary_Models", "Away_Win", 
        paths['output_dir'], model_params
    )
    
    print("\nTraining stacked model...")
    
    try:
        # Create and train stacked model
        stacked_model = StackedFootballPredictor(meta_model_params)
        stacked_model.fit(X_train, y_home_train, y_away_train, binary_results['home_win'])
        
        # Make predictions
        y_home_pred, y_away_pred = stacked_model.predict(X_test)
        y_home_prob, y_away_prob = stacked_model.predict_proba(X_test)
        
        # Calculate and save metrics for both targets
        results_file = os.path.join(paths['output_dir'], "stacked_model_binary_results.txt")
        with open(results_file, 'w') as f:
            f.write("Stacked Model Binary Prediction Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Home wins results
            f.write("\nHome Win Predictions:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {accuracy_score(y_home_test, y_home_pred):.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_home_test, y_home_pred))
            
            # Away wins results
            f.write("\nAway Win Predictions:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {accuracy_score(y_away_test, y_away_pred):.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_away_test, y_away_pred))
        
        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Home wins confusion matrix
        sns.heatmap(confusion_matrix(y_home_test, y_home_pred), 
                   annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Home Win Predictions')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Away wins confusion matrix
        sns.heatmap(confusion_matrix(y_away_test, y_away_pred),
                   annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Away Win Predictions')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(paths['output_dir'], "stacked_model_binary_confusion_matrices.png"))
        plt.close()
        
        return stacked_model
        
    except Exception as e:
        print(f"Error in stacked model training: {str(e)}")
        raise
def main():
    try:
        # Load configuration
        config = get_config()
        
        # Train stacked model
        stacked_model = train_stacked_model(config)
        
        print("\nStacked model training completed. Results saved in output directory.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()