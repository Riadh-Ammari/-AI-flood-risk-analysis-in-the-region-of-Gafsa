import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FLOOD RISK CLASSIFICATION MODEL TRAINING")
print("=" * 80)

# Load ML dataset
print("\n1. Loading ML dataset...")
ml_data = pd.read_csv('features/ml_dataset.csv')
print(f"   Dataset shape: {ml_data.shape}")

# Remove rows with NaN in risk_category
ml_data = ml_data.dropna(subset=['risk_category'])
print(f"   After removing NaN: {ml_data.shape}")
print(f"   Columns: {ml_data.columns.tolist()}")

# Separate features and target
feature_cols = ['slope', 'aspect', 'curvature', 'roughness', 'twi', 'elev_class', 
                'dist_water', 'dist_waterways', 'building_density', 'landuse_risk']
X = ml_data[feature_cols]
y = ml_data['risk_category']

print(f"   Features: {feature_cols}")
print(f"   Target distribution:\n{y.value_counts()}")
print(f"   Target classes: {list(y.unique())}")

# Convert risk categories to numeric labels
risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
y_numeric = y.map(risk_mapping)

print(f"\n2. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Standardize features
print(f"\n3. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   Scaler saved to models/feature_scaler.pkl")

# Train Random Forest Classifier
print(f"\n4. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
print(f"   Training complete!")

# Random Forest predictions
rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_test = rf_model.predict(X_test_scaled)

# Random Forest performance
print(f"\n   Random Forest Performance:")
print(f"   Training accuracy: {accuracy_score(y_train, rf_pred_train):.4f}")
print(f"   Test accuracy: {accuracy_score(y_test, rf_pred_test):.4f}")
print(f"   Precision (weighted): {precision_score(y_test, rf_pred_test, average='weighted', zero_division=0):.4f}")
print(f"   Recall (weighted): {recall_score(y_test, rf_pred_test, average='weighted', zero_division=0):.4f}")
print(f"   F1 Score (weighted): {f1_score(y_test, rf_pred_test, average='weighted', zero_division=0):.4f}")

# Train Gradient Boosting Classifier
print(f"\n5. Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train_scaled, y_train)
print(f"   Training complete!")

# Gradient Boosting predictions
gb_pred_train = gb_model.predict(X_train_scaled)
gb_pred_test = gb_model.predict(X_test_scaled)

# Gradient Boosting performance
print(f"\n   Gradient Boosting Performance:")
print(f"   Training accuracy: {accuracy_score(y_train, gb_pred_train):.4f}")
print(f"   Test accuracy: {accuracy_score(y_test, gb_pred_test):.4f}")
print(f"   Precision (weighted): {precision_score(y_test, gb_pred_test, average='weighted', zero_division=0):.4f}")
print(f"   Recall (weighted): {recall_score(y_test, gb_pred_test, average='weighted', zero_division=0):.4f}")
print(f"   F1 Score (weighted): {f1_score(y_test, gb_pred_test, average='weighted', zero_division=0):.4f}")

# Cross-validation
print(f"\n6. Cross-validation (5-fold)...")
cv_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_gb = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"   Random Forest CV accuracy: {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")
print(f"   Gradient Boosting CV accuracy: {cv_gb.mean():.4f} (+/- {cv_gb.std():.4f})")

# Save models
print(f"\n7. Saving trained models...")
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('models/gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
print(f"   Models saved!")

# Feature importance
print(f"\n8. Feature Importance Analysis:")
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"   Random Forest:")
print(rf_importance.to_string(index=False))

gb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n   Gradient Boosting:")
print(gb_importance.to_string(index=False))

# Generate visualizations
print(f"\n9. Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Training Results - Random Forest vs Gradient Boosting', fontsize=16, fontweight='bold', y=1.00)

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_pred_test)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Random Forest - Confusion Matrix', fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_xticklabels(['Low', 'Med', 'High', 'V.High'], rotation=45)
axes[0, 0].set_yticklabels(['Low', 'Med', 'High', 'V.High'], rotation=45)

# Confusion Matrix - Gradient Boosting
cm_gb = confusion_matrix(y_test, gb_pred_test)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Gradient Boosting - Confusion Matrix', fontweight='bold')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_xticklabels(['Low', 'Med', 'High', 'V.High'], rotation=45)
axes[0, 1].set_yticklabels(['Low', 'Med', 'High', 'V.High'], rotation=45)

# Feature Importance - Random Forest
axes[0, 2].barh(rf_importance['feature'], rf_importance['importance'], color='steelblue')
axes[0, 2].set_xlabel('Importance')
axes[0, 2].set_title('Random Forest - Feature Importance', fontweight='bold')
axes[0, 2].invert_yaxis()

# Feature Importance - Gradient Boosting
axes[1, 0].barh(gb_importance['feature'], gb_importance['importance'], color='darkgreen')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Gradient Boosting - Feature Importance', fontweight='bold')
axes[1, 0].invert_yaxis()

# Model Performance Comparison
models = ['Random\nForest', 'Gradient\nBoosting']
train_acc = [accuracy_score(y_train, rf_pred_train), accuracy_score(y_train, gb_pred_train)]
test_acc = [accuracy_score(y_test, rf_pred_test), accuracy_score(y_test, gb_pred_test)]
x_pos = np.arange(len(models))
width = 0.35
axes[1, 1].bar(x_pos - width/2, train_acc, width, label='Train', color='skyblue')
axes[1, 1].bar(x_pos + width/2, test_acc, width, label='Test', color='coral')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Model Accuracy Comparison', fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].set_ylim([0.7, 1.0])

# Detailed metrics
metrics_data = {
    'Random Forest': [
        accuracy_score(y_test, rf_pred_test),
        precision_score(y_test, rf_pred_test, average='weighted', zero_division=0),
        recall_score(y_test, rf_pred_test, average='weighted', zero_division=0),
        f1_score(y_test, rf_pred_test, average='weighted', zero_division=0)
    ],
    'Gradient Boosting': [
        accuracy_score(y_test, gb_pred_test),
        precision_score(y_test, gb_pred_test, average='weighted', zero_division=0),
        recall_score(y_test, gb_pred_test, average='weighted', zero_division=0),
        f1_score(y_test, gb_pred_test, average='weighted', zero_division=0)
    ]
}
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_metrics = np.arange(len(metrics_names))
width_metrics = 0.35
axes[1, 2].bar(x_metrics - width_metrics/2, metrics_data['Random Forest'], width_metrics, label='Random Forest', color='steelblue')
axes[1, 2].bar(x_metrics + width_metrics/2, metrics_data['Gradient Boosting'], width_metrics, label='Gradient Boosting', color='darkgreen')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Detailed Performance Metrics (Test Set)', fontweight='bold')
axes[1, 2].set_xticks(x_metrics)
axes[1, 2].set_xticklabels(metrics_names, rotation=45, ha='right')
axes[1, 2].set_ylim([0.7, 1.0])
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('outputs/model_training_results.png', dpi=300, bbox_inches='tight')
print(f"   Saved: outputs/model_training_results.png")
plt.close()

# Classification reports
print(f"\n10. Classification Reports:")
print(f"\n{'='*60}")
print(f"RANDOM FOREST CLASSIFIER")
print(f"{'='*60}")
print(classification_report(y_test, rf_pred_test, 
                          target_names=['Low', 'Medium', 'High', 'Very High'],
                          zero_division=0))

print(f"\n{'='*60}")
print(f"GRADIENT BOOSTING CLASSIFIER")
print(f"{'='*60}")
print(classification_report(y_test, gb_pred_test, 
                          target_names=['Low', 'Medium', 'High', 'Very High'],
                          zero_division=0))

print(f"\n{'='*80}")
print(f"MODEL TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\nNext step: Flood susceptibility mapping using trained models")
