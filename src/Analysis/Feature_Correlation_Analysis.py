import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('processed_matches_for_prediction.csv')

# Select only numerical features
exclude_cols = ['date', 'home_team', 'away_team', 'target_result', 'target_home_goals', 'target_away_goals']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Calculate correlation matrix
correlation_matrix = df[feature_cols].corr()

# Create correlation analysis summary
def analyze_correlations(correlation_matrix, threshold=0.5):
    # Get pairs of highly correlated features
    high_correlations = []
    features = correlation_matrix.columns
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                high_correlations.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr
                })
    
    # Sort by absolute correlation value
    high_correlations = sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    return high_correlations

# Analyze home team features correlations
home_features = [col for col in feature_cols if col.startswith('home_')]
home_correlation_matrix = df[home_features].corr()
home_correlations = analyze_correlations(home_correlation_matrix)

# Analyze away team features correlations
away_features = [col for col in feature_cols if col.startswith('away_')]
away_correlation_matrix = df[away_features].corr()
away_correlations = analyze_correlations(away_correlation_matrix)

# Print summary of high correlations
print("High Correlations for Home Team Features:")
for corr in home_correlations[:10]:  # Top 10 correlations
    print(f"{corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")

print("\nHigh Correlations for Away Team Features:")
for corr in away_correlations[:10]:  # Top 10 correlations
    print(f"{corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")

# Create correlation plots
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Calculate most important features for predicting goals
target_correlations = pd.DataFrame({
    'home_goals_correlation': df[feature_cols].corrwith(df['target_home_goals']).abs(),
    'away_goals_correlation': df[feature_cols].corrwith(df['target_away_goals']).abs()
}).sort_values(by=['home_goals_correlation', 'away_goals_correlation'], ascending=[False, False])

print("\nTop Features Correlated with Target Home Goals:")
print(target_correlations.head(10)['home_goals_correlation'])

print("\nTop Features Correlated with Target Away Goals:")
print(target_correlations.head(10)['away_goals_correlation'])

# Save detailed correlation analysis to CSV
correlation_matrix.to_csv('feature_correlations.csv')
target_correlations.to_csv('target_correlations.csv')