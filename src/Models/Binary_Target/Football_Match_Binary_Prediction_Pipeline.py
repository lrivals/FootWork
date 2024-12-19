import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.Config.Config_Manager import ConfigManager

def get_models(config):
    """Initialize models with parameters from config"""
    model_params = config.get_config_value('model_parameters', default={})
    
    # Get CatBoost params and add train_dir
    catboost_params = model_params.get('catboost', {})
    catboost_params['train_dir'] = 'src/Models/Binary_Target/catboost_info'
    
    models = {
        'Random Forest': RandomForestClassifier(**model_params.get('random_forest', {})),
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'SVM': SVC(**model_params.get('svm', {})),
        'Gradient Boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        'XGBoost': XGBClassifier(**model_params.get('xgboost', {})),
        'LightGBM': LGBMClassifier(**model_params.get('lightgbm', {})),
        'CatBoost': CatBoostClassifier(**catboost_params),
        #'Neural Network': MLPClassifier(**model_params.get('neural_network', {})), 
        'KNN': KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees': ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }
    return models

def load_and_prepare_binary_data(config):
    input_path = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])
    split_params = config.get_config_value('data_split', default={'test_size': 0.2, 'random_state': 42})
    
    df = pd.read_csv(input_path)
    
    # Create both binary targets
    df['home_win'] = (df['target_result'] == 'HomeWin').astype(int)
    df['away_win'] = (df['target_result'] == 'AwayWin').astype(int)
    
    X = df.drop(exclude_columns + ['target_result', 'home_win', 'away_win'], axis=1)
    y_home = df['home_win']
    y_away = df['away_win']
    
    # Split data for both targets
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, **split_params
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_home_train, y_home_test, y_away_train, y_away_test

def save_results(results, dataset_name, target_type, config):
    """Save metrics to file"""
    output_dir = config.get_paths()['output_dir']
    filename = os.path.join(output_dir, f"metrics_results_{dataset_name}_{target_type}.txt")
    
    with open(filename, 'w') as f:
        f.write(f"Results for {dataset_name} - {target_type}\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"ROC AUC: {result.get('roc_auc', 'N/A')}\n\n")
            f.write("Classification Report:\n")
            f.write(result['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['confusion_matrix']))
            f.write("\n" + "=" * 50 + "\n")

def plot_roc_curves(models_dict, X_test, y_test, dataset_name, target_type, output_dir):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in models_dict.items():
        try:
            model = result['model']
            y_score = (model.decision_function(X_test) if hasattr(model, 'decision_function')
                      else model.predict_proba(X_test)[:, 1])
            
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            result['roc_auc'] = roc_auc
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        except (AttributeError, NotImplementedError) as e:
            print(f"Warning: Could not generate ROC curve for {model_name}: {str(e)}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name} ({target_type})')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(output_dir, f"roc_curves_{dataset_name}_{target_type}.png"),
                bbox_inches='tight', dpi=300)
    plt.close()
    
def train_all_models(config):
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = load_and_prepare_binary_data(config)
    output_dir = config.get_paths()['output_dir']
    models = get_models(config)
    
    home_results = {}
    away_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train and evaluate for home wins
            home_results[name] = train_and_evaluate_model(
                model, X_train, X_test, y_home_train, y_home_test,
                'Home Win', 'Full_Dataset', output_dir
            )
            
            # Create new instance for away wins
            model_away = type(model)(**model.get_params())
            away_results[name] = train_and_evaluate_model(
                model_away, X_train, X_test, y_away_train, y_away_test,
                'Away Win', 'Full_Dataset', output_dir
            )
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Plot ROC curves
    plot_roc_curves(home_results, X_test, y_home_test, "Full_Dataset", "Home Win", output_dir)
    plot_roc_curves(away_results, X_test, y_away_test, "Full_Dataset", "Away Win", output_dir)
    
    return home_results, away_results

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_type, 
                           dataset_name, output_dir):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred,
                                                    target_names=[f'Not {target_type}', target_type])
    }
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Not {target_type}', target_type],
                yticklabels=[f'Not {target_type}', target_type])
    plt.title(f'Confusion Matrix - {model.__class__.__name__} ({dataset_name} - {target_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 
        f"confusion_matrix_{dataset_name}_{target_type}_{model.__class__.__name__}.png"))
    plt.close()
    
    return results

def save_all_results(home_results, away_results, config):
    output_dir = config.get_paths()['output_dir']
    
    for target_type, results in [("Home_Win", home_results), ("Away_Win", away_results)]:
        filename = os.path.join(output_dir, f"metrics_results_Full_Dataset_{target_type}.txt")
        with open(filename, 'w') as f:
            f.write(f"Results for Full Dataset - {target_type}\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"\nModel: {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"ROC AUC: {result.get('roc_auc', 'N/A')}\n\n")
                f.write("Classification Report:\n")
                f.write(result['classification_report'])
                f.write("\nConfusion Matrix:\n")
                f.write(str(result['confusion_matrix']))
                f.write("\n" + "=" * 50 + "\n")

def ensure_directories(config):
    """Ensure all necessary directories exist"""
    output_dir = config.get_paths()['output_dir']
    model_dirs = [
        Path(output_dir),
        Path('src/Models/Binary_Target/catboost_info')
    ]
    for dir_path in model_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        
def main():
    try:
        config = ConfigManager('src/Config/configBT_1.yaml')
        ensure_directories(config)  # Add this line
        home_results, away_results = train_all_models(config)
        save_all_results(home_results, away_results, config)
        print("\nTraining completed. Results saved in output directory.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()