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
sys.path.append(str(Path(__file__).parent.parent))
from Binary_Target.Football_Match_Binary_Prediction_Pipeline import ConfigManager

class StackedFootballPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_model_params):
        """Initialize stacked predictor"""
        self.meta_model = RandomForestClassifier(**meta_model_params)
        self.binary_models = {}
        self.le = LabelEncoder()
    
    def fit(self, X, y, binary_results):
        """
        Fit the stacked model using binary classifiers' predictions
        """
        # Store binary models
        self.binary_models = {
            name: results['model'] 
            for name, results in binary_results.items()
        }
        
        # Generate meta-features
        meta_features = self._get_meta_features(X)
        
        # Fit meta-model
        self.le.fit(y)
        y_encoded = self.le.transform(y)
        self.meta_model.fit(meta_features, y_encoded)
        
        return self
    
    def _get_meta_features(self, X):
        """Generate meta-features from binary model predictions"""
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.binary_models.values()
        ])
        return meta_features
    
    def predict(self, X):
        """Make predictions using the stacked model"""
        meta_features = self._get_meta_features(X)
        predictions = self.meta_model.predict(meta_features)
        return self.le.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

def get_config():
    """Get configuration with validation"""
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent.parent / 'src' / 'Config' / 'configBT_Stacked1.yaml'
    
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
    """Train and evaluate the stacked model"""
    print("\nAvailable config sections:")
    for key in config.config.keys():
        print(f"- {key}")
    
    paths = config.get_paths()
    exclude_columns = config.get_excluded_columns()
    split_params = config.get_split_params()
    model_params = config.get_model_params()
    
    try:
        meta_model_params = config.config['meta_classifier']['random_forest']
        print("\nUsing meta-classifier parameters:", meta_model_params)
    except KeyError as e:
        print(f"Error: Could not find meta-classifier parameters in config file. Error: {e}")
        print("Please ensure your config file has a 'meta_classifier' section with 'random_forest' parameters.")
        raise
    
    # Prepare the full dataset
    print("\nPreparing full dataset...")
    X_full, y_full, y_home, y_away = prepare_full_dataset(paths['full_dataset'], exclude_columns)
    
    # Split the full dataset
    X_train, X_test, y_train, y_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X_full, y_full, y_home, y_away, **split_params
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
        stacked_model.fit(X_train, y_train, binary_results['home_win'])
        
        # Make predictions
        y_pred = stacked_model.predict(X_test)
        y_prob = stacked_model.predict_proba(X_test)
        
        # Perform cross-validation on stacked model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        print("\nPerforming cross-validation on stacked model...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            # Split data for this fold
            X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
            y_fold_train, y_fold_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
            
            # Train binary models for this fold
            fold_binary_results = {}
            for target_type, y_binary in [('home_win', y_home), ('away_win', y_away)]:
                y_binary_fold_train = y_binary.iloc[train_idx]
                fold_results = {}
                for name, params in model_params.items():
                    if name == 'random_forest':
                        model = RandomForestClassifier(**params)
                    elif name == 'logistic_regression':
                        model = LogisticRegression(**params)
                    else:
                        model = SVC(**params)
                    model.fit(X_fold_train, y_binary_fold_train)
                    fold_results[name] = {'model': model}
                fold_binary_results[target_type] = fold_results
            
            # Train and evaluate stacked model for this fold
            fold_stacked = StackedFootballPredictor(meta_model_params)
            fold_stacked.fit(X_fold_train, y_fold_train, fold_binary_results['home_win'])
            fold_pred = fold_stacked.predict(X_fold_val)
            cv_scores.append(accuracy_score(y_fold_val, fold_pred))
            
            print(f"Fold {fold + 1} accuracy: {cv_scores[-1]:.4f}")
        
        # Calculate and save metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Save results
        results_file = os.path.join(paths['output_dir'], "stacked_model_results.txt")
        with open(results_file, 'w') as f:
            f.write("Stacked Model Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Architecture:\n")
            f.write("- Binary Models: Random Forest, Logistic Regression, SVM\n")
            f.write("- Meta Model: Random Forest\n\n")
            
            f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
            f.write("Cross-validation Results:\n")
            f.write(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}\n")
            f.write(f"Std CV Accuracy: {np.std(cv_scores):.4f}\n")
            f.write("Individual fold scores:\n")
            for i, score in enumerate(cv_scores):
                f.write(f"Fold {i+1}: {score:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write(class_report)
            
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
            
            f.write("\nClass Distribution:\n")
            for label, count in zip(*np.unique(y_pred, return_counts=True)):
                f.write(f"{label}: {count}\n")
            
            f.write("\nPrediction Probabilities Summary:\n")
            prob_summary = pd.DataFrame(y_prob, columns=stacked_model.le.classes_).describe()
            f.write(str(prob_summary))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_full),
                    yticklabels=np.unique(y_full))
        plt.title('Confusion Matrix - Stacked Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(paths['output_dir'], "stacked_model_confusion_matrix.png"))
        plt.close()
        
        # Plot cross-validation results
        plt.figure(figsize=(8, 6))
        plt.boxplot(cv_scores)
        plt.title('Cross-validation Scores - Stacked Model')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(paths['output_dir'], "stacked_model_cv_scores.png"))
        plt.close()
        
        # Plot probability distributions
        plt.figure(figsize=(10, 6))
        for i, class_name in enumerate(stacked_model.le.classes_):
            sns.kdeplot(y_prob[:, i], label=class_name)
        plt.title("Prediction Probability Distributions")
        plt.xlabel("Probability")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(paths['output_dir'], "stacked_model_probability_distributions.png"))
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