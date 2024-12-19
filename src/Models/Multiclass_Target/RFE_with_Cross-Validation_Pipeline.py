import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from src.Config.Config_Manager import ConfigManager

def get_models(config):
    model_params = config.get_config_value('model_parameters', default={})
    
    # Get CatBoost params and add train_dir
    catboost_params = model_params.get('catboost', {})
    catboost_params['train_dir'] = 'src/Models/Multiclass_Target/catboost_info'
    
    # Get XGBoost params and add necessary parameters
    xgboost_params = model_params.get('xgboost', {})
    xgboost_params.update({
        'use_label_encoder': False,
        'enable_categorical': True,
        'objective': 'multi:softmax',
        'num_class': 3  # for HomeWin, Draw, AwayWin
    })
    
    return {
        #'Random Forest': RandomForestClassifier(**model_params.get('random_forest', {})),
        #'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        #'Gradient Boosting': GradientBoostingClassifier(**model_params.get('gradient_boosting', {})),
        #'XGBoost': XGBClassifier(**xgboost_params),
        #'LightGBM': LGBMClassifier(**model_params.get('lightgbm', {})),
        #'CatBoost': CatBoostClassifier(**catboost_params),
        #'Neural Network': MLPClassifier(**model_params.get('neural_network', {})),
        #'KNN': KNeighborsClassifier(**model_params.get('knn', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {})),
        'Extra Trees': ExtraTreesClassifier(**model_params.get('extra_trees', {}))
    }

def perform_rfecv(model, X, y, cv=5):
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=StratifiedKFold(cv),
        scoring='accuracy',
        n_jobs=-1
    )
    rfecv.fit(X, y)
    return rfecv

def plot_rfecv_results(rfecv, model_name, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Get scores and optimal values
    scores = rfecv.cv_results_['mean_test_score']
    optimal_n_features = rfecv.n_features_
    max_score = max(scores)
    
    # Plot scores
    plt.plot(range(1, len(scores) + 1), scores, 'o-')
    
    # Add vertical line at optimal number of features
    plt.axvline(x=optimal_n_features, color='red', linestyle='--', alpha=0.5)
    plt.text(optimal_n_features + 0.2, plt.ylim()[0], f'Optimal features: {optimal_n_features}', 
             rotation=90, verticalalignment='bottom')
    
    # Add horizontal line at maximum score
    plt.axhline(y=max_score, color='green', linestyle='--', alpha=0.5)
    plt.text(plt.xlim()[0], max_score + 0.001, f'Max accuracy: {max_score:.3f}', 
             verticalalignment='bottom')
    
    plt.grid(True)
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-validation Accuracy')
    plt.title(f'Feature Selection - {model_name}')
    plt.savefig(os.path.join(output_dir, f'rfecv_scores_{model_name.replace(" ", "_")}.png'))
    plt.close()

def save_selected_features(X, rfecv, model_name, output_dir):
    feature_names = X.columns
    selected_features = feature_names[rfecv.support_]
    
    # Get min and max accuracy scores and their corresponding number of features
    scores = rfecv.cv_results_['mean_test_score']
    max_score = max(scores)
    min_score = min(scores)
    max_features_count = np.where(scores == max_score)[0][0] + 1  # Add 1 because we count from 1
    min_features_count = np.where(scores == min_score)[0][0] + 1  # Add 1 because we count from 1
    
    with open(os.path.join(output_dir, f'selected_features_{model_name.replace(" ", "_")}.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimal number of features: {rfecv.n_features_}\n")
        f.write(f"Highest CV accuracy: {max_score:.4f} (with {max_features_count} features)\n")
        f.write(f"Lowest CV accuracy: {min_score:.4f} (with {min_features_count} features)\n\n")
        f.write("Selected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
    
    return {
        'model_name': model_name,
        'optimal_features': rfecv.n_features_,
        'max_score': max_score,
        'max_score_features': max_features_count,
        'min_score': min_score,
        'min_score_features': min_features_count,
        'test_scores': None  # Will be updated after evaluation
    }

def prepare_and_save_optimal_dataset(X, y, rfecv, model_name, config):
    selected_features = X.columns[rfecv.support_]
    optimal_dataset = pd.concat([X[selected_features], y], axis=1)
    
    output_dir = config.get_paths()['output_dir']
    output_path = os.path.join(output_dir, f'optimal_dataset_{model_name.replace(" ", "_")}.csv')
    optimal_dataset.to_csv(output_path, index=False)
    return X[selected_features]

def evaluate_model(model, X_train, X_test, y_train, y_test, class_names, model_name, output_dir):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} (Optimal Features)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    return accuracy, class_report

def save_all_metrics(all_results, output_dir):
    with open(os.path.join(output_dir, 'all_models_metrics.txt'), 'w') as f:
        f.write("Feature Selection and Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Sort models by test accuracy
        sorted_results = sorted(all_results, key=lambda x: x['test_scores']['accuracy'], reverse=True)
        
        # Write summary table
        f.write("Summary Table:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model Name':<20} {'Test Acc':<10} {'Best CV Acc':<12} {'#Features':<10} {'Worst CV Acc':<12} {'#Features':<10}\n")
        f.write("-" * 100 + "\n")
        
        for result in sorted_results:
            f.write(f"{result['model_name']:<20} "
                   f"{result['test_scores']['accuracy']:.4f}     "
                   f"{result['max_score']:.4f}      "
                   f"{result['max_score_features']:<10} "
                   f"{result['min_score']:.4f}      "
                   f"{result['min_score_features']:<10}\n")
        
        f.write("\n\n")
        f.write("Detailed Results by Model:\n")
        f.write("=" * 50 + "\n\n")
        
        # Write detailed results for each model
        for result in sorted_results:
            f.write(f"Model: {result['model_name']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Optimal number of features: {result['optimal_features']}\n")
            f.write(f"Best CV accuracy: {result['max_score']:.4f} (with {result['max_score_features']} features)\n")
            f.write(f"Worst CV accuracy: {result['min_score']:.4f} (with {result['min_score_features']} features)\n")
            f.write(f"Test accuracy: {result['test_scores']['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result['test_scores']['classification_report'])
            f.write("\n" + "=" * 50 + "\n\n")

def ensure_directories(config):
    """Ensure all necessary directories exist"""
    output_dir = config.get_paths()['output_dir']
    model_dirs = [
        Path(output_dir),
        Path('src/Models/Multiclass_Target/catboost_info')  # For consistency with other script
    ]
    for dir_path in model_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def main():
    try:
        config = ConfigManager('src/Config/configFS_1.yaml')
        ensure_directories(config)
        
        # Load data
        df = pd.read_csv(config.get_paths()['full_dataset'])
        exclude_columns = config.get_config_value('excluded_columns', default=[])
        X = df.drop(exclude_columns + ['target_result'], axis=1)
        y = df['target_result']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Perform RFECV for each model
        models = get_models(config)
        output_dir = config.get_paths()['output_dir']
        
        all_results = []
        for name, model in models.items():
            print(f"\nProcessing {name}...")
            
            # Perform RFECV
            rfecv = perform_rfecv(model, X_scaled, y_encoded)
            
            # Plot and save results
            plot_rfecv_results(rfecv, name, output_dir)
            result_dict = save_selected_features(X_scaled, rfecv, name, output_dir)
            
            # Prepare optimal dataset
            X_optimal = prepare_and_save_optimal_dataset(X_scaled, y, rfecv, name, config)
            
            # Split and evaluate
            X_train, X_test, y_train, y_test = train_test_split(
                X_optimal, y_encoded,
                test_size=0.2, random_state=42
            )
            
            accuracy, class_report = evaluate_model(model, X_train, X_test, y_train, y_test, 
                                                class_names, name, output_dir)
            
            # Store test results
            result_dict['test_scores'] = {
                'accuracy': accuracy,
                'classification_report': class_report
            }
            all_results.append(result_dict)

        # Save all metrics in one file
        save_all_metrics(all_results, output_dir)
        print("\nFeature selection and evaluation completed.")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()