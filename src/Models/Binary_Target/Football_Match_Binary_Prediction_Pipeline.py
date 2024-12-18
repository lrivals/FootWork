import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yaml
from pathlib import Path
import sys

# Add the parent directory to the Python path to import the binary prediction module
sys.path.append(str(Path(__file__).parent.parent.parent))
from Config.Config_Manager import ConfigManager

def load_and_prepare_binary_data(input_path, exclude_columns, target_type='home', split_params=None):
    """
    Load and prepare the dataset for binary classification
    """
    # Read the data
    df = pd.read_csv(input_path)
    
    # Print value counts before processing
    print(f"\nOriginal target distribution:")
    print(df['target_result'].value_counts())
    
    # Create binary targets
    if target_type == 'home':
        df['binary_target'] = (df['target_result'] == 'HomeWin').astype(int)
        target_name = 'Home Win'
    else:
        df['binary_target'] = (df['target_result'] == 'AwayWin').astype(int)
        target_name = 'Away Win'
    
    # Print binary target distribution
    print(f"\nBinary target distribution for {target_type}:")
    print(df['binary_target'].value_counts())
    
    # Create feature matrix X and target vector y
    X = df.drop(exclude_columns + ['target_result', 'binary_target'], axis=1)
    y = df['binary_target']
    
    # Verify we have both classes
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Only found classes {unique_classes} in the dataset. Need both 0 and 1.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **split_params
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, target_name

def save_results_to_file(results, dataset_name, target_type, output_dir):
    """Save metrics results to a text file"""
    filename = os.path.join(output_dir, f"metrics_results_{dataset_name}_{target_type}.txt")
    
    with open(filename, 'w') as f:
        f.write(f"Results for {dataset_name} - {target_type}\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"ROC AUC: {result['roc_auc']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['confusion_matrix']))
            f.write("\n" + "=" * 50 + "\n")

def plot_roc_curve(models_dict, X_test, y_test, dataset_name, target_type, output_dir):
    """Plot ROC curve for binary classification"""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in models_dict.items():
        model = result['model']
        
        # Get probability predictions
        if isinstance(model, SVC):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Store ROC AUC
        result['roc_auc'] = roc_auc
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name} ({target_type})')
    plt.legend(loc="lower right")
    
    # Save ROC curves plot
    roc_filename = os.path.join(output_dir, f"roc_curves_{dataset_name}_{target_type}.png")
    plt.savefig(roc_filename, bbox_inches='tight', dpi=300)
    plt.close()

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name, 
                            target_type, output_dir, model_params):
    """Train and evaluate multiple models"""
    # Initialize models with parameters from config
    models = {
        'Random Forest': RandomForestClassifier(**model_params['random_forest']),
        'Logistic Regression': LogisticRegression(**model_params['logistic_regression']),
        'SVM': SVC(**model_params['svm'])
    }
    
    results = {}
    class_labels = ['Not ' + target_type, target_type]
    
    for name, model in models.items():
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.title(f'Confusion Matrix - {name} ({dataset_name} - {target_type})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix plot
        plot_filename = os.path.join(output_dir, 
            f"confusion_matrix_{dataset_name}_{target_type}_{name.replace(' ', '_')}.png")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    # Plot ROC curves for all models
    plot_roc_curve(results, X_test, y_test, dataset_name, target_type, output_dir)
    
    return results

def main():
    # Load configuration
    config = ConfigManager()
    paths = config.get_paths()
    exclude_columns = config.get_excluded_columns()
    model_params = config.get_model_params()
    split_params = config.get_split_params()
    
    # Process both target types for full dataset
    for target_type in ['home', 'away']:
        try:
            print(f"\nProcessing full dataset for {target_type} wins...")
            X_train, X_test, y_train, y_test, feature_names, target_name = load_and_prepare_binary_data(
                paths['full_dataset'], exclude_columns, target_type, split_params
            )
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, 
                "Full_Dataset", target_name, paths['output_dir'], model_params
            )
            save_results_to_file(results, "Full_Dataset", target_name, paths['output_dir'])
        except Exception as e:
            print(f"Error processing {target_type} wins: {str(e)}")
    
    # Process both target types for PCA dataset
    for target_type in ['home', 'away']:
        try:
            print(f"\nProcessing PCA dataset for {target_type} wins...")
            X_train, X_test, y_train, y_test, feature_names, target_name = load_and_prepare_binary_data(
                paths['pca_dataset'], [], target_type, split_params
            )
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, 
                "PCA_Dataset", target_name, paths['output_dir'], model_params
            )
            save_results_to_file(results, "PCA_Dataset", target_name, paths['output_dir'])
        except Exception as e:
            print(f"Error processing {target_type} wins: {str(e)}")

if __name__ == "__main__":
    main()