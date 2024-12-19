# Analysis of Football Match Prediction Models

## Overall Performance Analysis

The models were evaluated on a dataset with an imbalanced distribution:
- HomeWin: 2248 samples (45.8%)
- AwayWin: 1407 samples (28.7%)
- Draw: 1250 samples (25.5%)

### Top Performing Models

1. **Gradient Boosting**: 52.52% accuracy
   - Best overall accuracy
   - Strong precision for away wins (0.50)
   - Highest home win recall (0.84)
   - However, very poor on draws (0.03 recall)

2. **Logistic Regression**: 52.15% accuracy
   - Second-best performer
   - Balanced precision across classes
   - Similar pattern of poor draw prediction

3. **CatBoost**: 52.44% accuracy
   - Highest draw precision (0.58)
   - Best home win recall (0.85)
   - Most consistent precision scores

### Key Observations

1. **Class Imbalance Impact**
   - All models struggle with predicting draws
   - Most models show bias toward predicting home wins
   - Best draw recall is achieved by SVM (0.35)

2. **Model Behaviors**
   - Traditional ML models (Gradient Boosting, Logistic Regression) outperform deep learning
   - Neural Network shows most balanced predictions but lowest accuracy (41.39%)
   - Ensemble methods generally perform better than single models

3. **Precision vs Recall Trade-offs**
   - Home wins: Generally high recall (0.81-0.85) but moderate precision (0.53-0.55)
   - Away wins: Moderate precision and recall (around 0.45-0.50)
   - Draws: Very poor recall across all models except SVM

## Model-Specific Analysis

### Strong Performers

1. **Gradient Boosting**
   ```
   Accuracy: 0.5252
   Strengths: Best overall accuracy, good home win predictions
   Weaknesses: Poor draw predictions
   ```

2. **CatBoost**
   ```
   Accuracy: 0.5244
   Strengths: Most balanced precision, excellent home win recall
   Weaknesses: Very low draw recall
   ```

### Moderate Performers

1. **XGBoost & LightGBM**
   ```
   Accuracy: ~0.518
   Notable: More balanced predictions but lower overall accuracy
   ```

2. **Random Forest & Extra Trees**
   ```
   Accuracy: ~0.511-0.513
   Notable: Consistent performance across metrics
   ```

### Poor Performers

1. **Neural Network**
   ```
   Accuracy: 0.4139
   Notable: Most balanced predictions but lowest accuracy
   ```

2. **KNN**
   ```
   Accuracy: 0.4532
   Notable: Struggles with class imbalance
   ```

## Recommendations

1. **Model Selection**
   - Use Gradient Boosting or CatBoost for best overall performance
   - Consider SVM if draw predictions are important
   - Avoid Neural Networks for this specific problem

2. **Improvement Strategies**
   - Address class imbalance through sampling techniques
   - Consider binary classification (Home vs Away) given poor draw predictions
   - Experiment with feature engineering and selection
   - Try ensemble stacking focusing on the top performers

3. **Deployment Considerations**
   - Monitor home win bias in production
   - Implement probability calibration for better prediction reliability
   - Consider cost-sensitive learning approaches

## Technical Limitations

The current models show several limitations:
- Overall accuracy ceiling around 53%
- Systematic bias against draw predictions
- Strong tendency to predict home wins
- Limited generalization across different match conditions

These results suggest that while the models can predict better than random chance (33.3%), there's significant room for improvement in prediction accuracy and class balance.
