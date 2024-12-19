import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
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
    return {
        'Logistic Regression': LogisticRegression(**model_params.get('logistic_regression', {})),
        'AdaBoost': AdaBoostClassifier(**model_params.get('adaboost', {}))
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
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
             rfecv.cv_results_['mean_test_score'], 'o-')
    plt.grid(True)
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-validation Accuracy')
    plt.title(f'Feature Selection - {model_name}')
    plt.savefig(os.path.join(output_dir, f'rfecv_scores_{model_name.replace(" ", "_")}.png'))
    plt.close()

def save_selected_features(X, rfecv, model_name, output_dir):
    feature_names = X.columns
    selected_features = feature_names[rfecv.support_]
    
    with open(os.path.join(output_dir, f'selected_features_{model_name.replace(" ", "_")}.txt'), 'w') as f:
        f.write(f"Optimal number of features: {rfecv.n_features_}\n")
        f.write("\nSelected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")

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
    
    # Save metrics
    with open(os.path.join(output_dir, f'metrics_{model_name.replace(" ", "_")}.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
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

def main():
    try:
        config = ConfigManager('src/Config/configFS_1.yaml')
        
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
        
        for name, model in models.items():
            print(f"\nProcessing {name}...")
            
            # Perform RFECV
            rfecv = perform_rfecv(model, X_scaled, y_encoded)
            
            # Plot and save results
            plot_rfecv_results(rfecv, name, output_dir)
            save_selected_features(X_scaled, rfecv, name, output_dir)
            
            # Prepare optimal dataset
            X_optimal = prepare_and_save_optimal_dataset(X_scaled, y, rfecv, name, config)
            
            # Split and evaluate
            X_train, X_test, y_train, y_test = train_test_split(
                X_optimal, y_encoded,
                test_size=0.2, random_state=42
            )
            
            evaluate_model(model, X_train, X_test, y_train, y_test, 
                         class_names, name, output_dir)
        
        print("\nFeature selection and evaluation completed.")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
