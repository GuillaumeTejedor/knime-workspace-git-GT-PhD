import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import shap
from sklearn.utils import resample
import xgboost

csv_classification = '../IO Files/Stratification/Clusters Statistical Study/classification_dataset.csv'
df_classification = pd.read_csv(csv_classification).drop(["ID_PATIENT", "ALSFRS-R_DATE"], axis=1).dropna(axis=0)

print(df_classification.head(n=5))

cluster_labels = np.unique(df_classification["CLUSTER_LABEL"])
n_clusters = len(cluster_labels)
df_classification = df_classification.drop("CLUSTER_LABEL", axis=1)

# Initialize variables
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=False)

for cluster_label in cluster_labels:
    print("#### IS CLUSTER " + str(cluster_label) + " #####")
    
    # Get target variable for the current cluster
    target = "IS_CLUSTER_" + str(cluster_label)
    yi = df_classification[target].astype(int)
    
    # Sampling to balance classes
    class_groups = [df_classification[yi == label] for label in yi.unique()]
    min_size = min(len(group) for group in class_groups)
    undersampled_groups = [resample(group, replace=False, n_samples=min_size) for group in class_groups]
    df_undersampled = pd.concat(undersampled_groups)
    X_resampled = df_undersampled.iloc[:, :-n_clusters]
    y_resampled = df_undersampled[target]
    
    # Initialize storage for SHAP values and feature matrix for visualization
    shap_values_all = []
    feature_matrix_all = []
    
    # Perform cross-validation
    for train_index, test_index in cv.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        
        # Train the classifier
        clf = xgboost.XGBClassifier().fit(X_train, y_train)
        
        # Compute SHAP values
        explainer = shap.Explainer(clf, X_test)
        shap_values = explainer(X_test)
        
        # Store SHAP values and the corresponding test feature matrix
        shap_values_all.append(shap_values.values)
        feature_matrix_all.append(X_test)
    
    # Concatenate SHAP values and feature matrices from all folds
    shap_values_all = np.vstack(shap_values_all)
    feature_matrix_all = pd.concat(feature_matrix_all, axis=0)
    
    # Wrap in a SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values_all,
        base_values=np.mean([sv.base_values for sv in shap_values], axis=0),
        data=feature_matrix_all.values,
        feature_names=feature_matrix_all.columns
    )
    
    import matplotlib.pyplot as plt

    # Initialize a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

    # SHAP Summary Plot (Distribution of Feature Importance)
    plt.sca(axes[0])  # Set current axis
    shap.summary_plot(shap_explanation.values, shap_explanation.data, feature_names=shap_explanation.feature_names, show=False)
    axes[0].set_title("SHAP Summary Plot")

    # SHAP Bar Plot (Global Feature Importance)
    plt.sca(axes[1])  # Set current axis
    shap.plots.bar(shap_explanation, show=False)
    axes[1].set_title("SHAP Bar Plot")

    # Display the final figure with both plots
    plt.tight_layout()
    plt.show()