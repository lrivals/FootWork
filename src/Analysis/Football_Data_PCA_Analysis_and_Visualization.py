import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load the data
input_path = 'clean_premiere_league_data/all_seasons_combined.csv'
output_path = 'clean_premiere_league_data/pca_results.csv'
plot_3d_path = 'clean_premiere_league_data/football_pca_3d.png'
plot_importance_path = 'clean_premiere_league_data/pca_importance.png'

def process_and_visualize_football_data(input_path, output_path, plot_3d_path, plot_importance_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the data
    print("Loading data from:", input_path)
    df = pd.read_csv(input_path)
    
    # Separate features for PCA
    exclude_columns = ['date', 'season','home_team', 'away_team', 'target_result', 
                      'target_home_goals', 'target_away_goals']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    print(f"Processing {len(feature_columns)} features...")
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    # Apply PCA with all components
    pca = PCA()
    pca_result = pca.fit_transform(scaled_features)
    
    # Create DataFrame with all PCA results
    pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=pca_columns
    )
    
    # Add target_result back to the PCA DataFrame
    pca_df['target_result'] = df['target_result']
    
    # Print explained variance ratio for first few components
    print("\nExplained variance ratio (first 3 components):")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"PC3: {pca.explained_variance_ratio_[2]:.3f}")
    print(f"Total: {sum(pca.explained_variance_ratio_[:3]):.3f}")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map for the results
    result_colors = {'HomeWin': 'green', 'Draw': 'blue', 'AwayWin': 'red'}
    
    # Plot each result category separately
    for result in result_colors:
        mask = pca_df['target_result'] == result
        ax.scatter(
            pca_df.loc[mask, 'PC1'],
            pca_df.loc[mask, 'PC2'],
            pca_df.loc[mask, 'PC3'],
            c=result_colors[result],
            label=result,
            alpha=0.6
        )
    
    # Add labels and title
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.title('3D PCA Visualization of Football Match Features')
    
    # Add legend with custom labels
    ax.legend(title="Match Results")
    plt.show()
    # Save the 3D plot
    print(f"\nSaving 3D plot to: {plot_3d_path}")
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create component importance plot
    plt.figure(figsize=(12, 6))
    
    # Cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot individual and cumulative explained variance
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, 'bo-', label='Individual')
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
            cumulative_variance_ratio, 'ro-', label='Cumulative')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Components Importance')
    plt.legend()
    plt.grid(True)
    
    # Add threshold lines
    plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% threshold')
    plt.axhline(y=0.95, color='y', linestyle='--', alpha=0.5, label='95% threshold')
    
    # Save the importance plot
    print(f"Saving importance plot to: {plot_importance_path}")
    plt.savefig(plot_importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the PCA DataFrame with all components
    print(f"Saving complete PCA results to: {output_path}")
    pca_df.to_csv(output_path, index=False)
    
    # Create and save feature importance DataFrame
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_columns
    )
    
    importance_path = output_path.replace('.csv', '_feature_importance.csv')
    feature_importance.to_csv(importance_path)
    print(f"Saving feature importance to: {importance_path}")
    
    return pca_df, pca

# Execute the analysis
pca_df, pca_model = process_and_visualize_football_data(input_path, output_path, plot_3d_path, plot_importance_path)
print("\nProcessing complete!")
