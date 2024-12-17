import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_and_prepare_data(input_path, exclude_columns):
    """
    Load and prepare the dataset for modeling
    """
    # Read the data
    df = pd.read_csv(input_path)
    
    # Create feature matrix X and target vector y
    X = df.drop(exclude_columns + ['target_result'], axis=1)
    y = df['target_result']
    
    # Get unique class labels
    class_labels = sorted(y.unique())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, class_labels

def save_results_to_file(results, dataset_name, output_path):
    """
    Save metrics results to a text file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}/metrics_results_{dataset_name}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Results for {dataset_name}\n")
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

def plot_roc_curves(models_dict, X_test, y_test, dataset_name, class_labels, output_path):
    """
    Plot ROC curves for all models
    """
    # Create figure for ROC curves
    plt.figure(figsize=(12, 8))
    
    # Convert y_test to binary format for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=class_labels)
    n_classes = len(class_labels)
    
    # Colors for different classes
    colors = cycle(['blue', 'red', 'green'])
    
    for model_name, result in models_dict.items():
        model = result['model']
        
        # Get probability predictions
        if isinstance(model, SVC):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        if isinstance(model, SVC):
            # For SVM, handle decision function output
            if n_classes == 2:
                fpr[0], tpr[0], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                roc_auc[0] = auc(fpr[0], tpr[0])
            else:
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            # For other models, handle probability output
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{model_name} (class {class_labels[i]}, AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name}')
    plt.legend(loc="lower right")
    
    # Save ROC curves plot
    roc_filename = f"{output_path}/roc_curves_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(roc_filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return roc_auc

def get_models():
    models = {
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced'
        ),
        'SVM': SVC(
            random_state=42, 
            probability=True,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1
        ),
        'XGBoost': XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ),
        'LightGBM': LGBMClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1
        ),
        'CatBoost': CatBoostClassifier(
            random_seed=42,
            iterations=100,
            learning_rate=0.1,
            verbose=False
        ),
        'Neural Network': MLPClassifier(
            random_state=42,
            hidden_layer_sizes=(100, 50),
            max_iter=1000
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'AdaBoost': AdaBoostClassifier(
            random_state=42,
            n_estimators=100
        ),
        'Extra Trees': ExtraTreesClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'
        )
    }
    return models

# Update your train_and_evaluate_models function to use the new models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name, class_labels, output_path):
    """
    Train and evaluate multiple models
    """
    # Get the enhanced set of models
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
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
            
            # Print results
            print(f"Results for {name} on {dataset_name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(results[name]['classification_report'])
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels,
                        yticklabels=class_labels)
            plt.title(f'Confusion Matrix - {name} ({dataset_name})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_filename = f"{output_path}/confusion_matrix_{dataset_name}_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Plot ROC curves for all models
    roc_auc = plot_roc_curves(results, X_test, y_test, dataset_name, class_labels, output_path)
    
    # Add ROC AUC scores to results
    for name in results:
        results[name]['roc_auc'] = roc_auc
    
    return results

def main():
    # Define paths and create output directory if it doesn't exist
    input_path = 'clean_premiere_league_data/all_seasons_combined.csv'
    input_path2 = 'clean_premiere_league_data/pca_results.csv'
    output_path = 'clean_premiere_league_data'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    exclude_columns = [
        'date', 'season', 'home_team', 'away_team', 'target_result',
        'target_home_goals', 'target_away_goals'
    ]
    
    # Process and evaluate full dataset
    print("\nProcessing full dataset...")
    X_train_full, X_test_full, y_train_full, y_test_full, feature_names, class_labels = load_and_prepare_data(
        input_path, exclude_columns
    )
    results_full = train_and_evaluate_models(
        X_train_full, X_test_full, y_train_full, y_test_full, 
        "Full_Dataset", class_labels, output_path
    )
    
    # Save full dataset results
    save_results_to_file(results_full, "Full_Dataset", output_path)
    
    # Process and evaluate PCA dataset
    print("\nProcessing PCA dataset...")
    X_train_pca, X_test_pca, y_train_pca, y_test_pca, pca_features, class_labels = load_and_prepare_data(
        input_path2, []  # No columns to exclude for PCA dataset
    )
    results_pca = train_and_evaluate_models(
        X_train_pca, X_test_pca, y_train_pca, y_test_pca, 
        "PCA_Dataset", class_labels, output_path
    )
    
    # Save PCA dataset results
    save_results_to_file(results_pca, "PCA_Dataset", output_path)
    
    # Compare results between datasets
    comparison_df = pd.DataFrame({
        'Full Dataset': [results_full[model]['accuracy'] for model in results_full],
        'PCA Dataset': [results_pca[model]['accuracy'] for model in results_pca]
    }, index=results_full.keys())
    
    # Plot and save comparison
    plt.figure(figsize=(10, 6))
    comparison_df.plot(kind='bar')
    plt.title('Model Accuracy Comparison: Full vs PCA Dataset')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_filename = f"{output_path}/accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(comparison_plot_filename, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()