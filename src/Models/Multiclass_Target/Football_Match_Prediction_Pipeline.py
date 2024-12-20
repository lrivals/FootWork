import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.Config.Config_Manager import ConfigManager

def get_models(config):
    model_params = config.get_config_value('model_parameters', default={})
    
    return {
        'Random Forest': RandomForestClassifier(**model_params.get('random_forest', {})),
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'SVM': SVC(**model_params.get('svm', {})),
        'Gradient Boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        'XGBoost': XGBClassifier(**model_params.get('xgboost', {})),
        'LightGBM': LGBMClassifier(**model_params.get('lightgbm', {})),
        'CatBoost': CatBoostClassifier(**model_params.get('catboost', {}), 
                              train_dir='src/Models/Multiclass_Target/catboost_info'),
        #'Neural Network': MLPClassifier(**model_params.get('neural_network', {})),
        'KNN': KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees': ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }

def load_and_prepare_data(config):
    input_path = config.get_paths()['full_dataset']
    exclude_columns = config.get_config_value('excluded_columns', default=[])
    split_params = config.get_config_value('data_split', default={'test_size': 0.2, 'random_state': 42})
    
    df = pd.read_csv(input_path)
    
    print("\nTarget distribution:")
    print(df['target_result'].value_counts())
    
    X = df.drop(exclude_columns + ['target_result'], axis=1)
    y = df['target_result']
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Store class names
    class_names = le.classes_
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, **split_params)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, class_names, 
                           dataset_name, output_dir):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred,
                                                    target_names=class_names)
    }
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model.__class__.__name__} ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 
        f"confusion_matrix_{dataset_name}_{model.__class__.__name__}.png"))
    plt.close()
    
    return results

def train_all_models(config):
    X_train, X_test, y_train, y_test, class_names = load_and_prepare_data(config)
    output_dir = config.get_paths()['output_dir']
    models = get_models(config)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            results[name] = train_and_evaluate_model(
                model, X_train, X_test, y_train, y_test,
                class_names, "Full_Dataset", output_dir
            )
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def save_results(results, config):
    output_dir = config.get_paths()['output_dir']
    filename = os.path.join(output_dir, "metrics_results_Full_Dataset_Multiclass.txt")
    
    with open(filename, 'w') as f:
        f.write("Results for Full Dataset - Multiclass Prediction\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['confusion_matrix']))
            f.write("\n" + "=" * 50 + "\n")

def main():
    try:
        config = ConfigManager('src/Config/configMC_1.yaml')
        catboost_dir = Path('src/Models/Multiclass_Target/catboost_info')
        catboost_dir.mkdir(parents=True, exist_ok=True)
        results = train_all_models(config)
        save_results(results, config)
        print("\nTraining completed. Results saved in output directory.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()