#!/usr/bin/env python
# coding: utf-8
"""
Example script for running BayesianRuleSet on the coupon dataset.

This script demonstrates how to:
1. Load and preprocess the coupon data
2. Train a BayesianRuleSet model
3. Evaluate the model's performance
4. Visualize the results

Please ensure you have the coupon_data.csv file in the data/ directory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, f1_score, accuracy_score
)

# Add the parent directory to the path so we can import the ruleset package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ruleset import BayesianRuleSet


def load_data():
    """Load the coupon dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'coupon_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    return df


def preprocess_data(df):
    """Preprocess the dataset for training."""
    # Check if the target column exists (assuming it's called 'Y' or similar)
    target_column = None
    for col in ['Y', 'y', 'target', 'class', 'label']:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        raise ValueError("Could not identify the target column in the dataset")
    
    # Extract features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"Target variable distribution:\n{y.value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    """Train and evaluate the BayesianRuleSet model."""
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Initialize the model with optimal parameters
    # These parameter values can be adjusted based on the dataset characteristics
    model = BayesianRuleSet(
        max_rules=3000,
        max_iter=30000,
        support=5,
        maxlen=3,
        alpha_1=15,
        beta_1=1,
        alpha_2=15,
        beta_2=1,
        level=4,
        method='forest',
        forest_size=300,
        propose_threshold=0.3,
        greedy_initilization=True
    )
    
    print("\n" + "="*50)
    print("Training BayesianRuleSet model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\n" + "="*50)
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print the final ruleset
    print("\n" + "="*50)
    print("Final Ruleset:")
    model.print_rules(model.predicted_rules)
    
    # Plot ROC curve if probabilistic predictions are available
    try:
        # Some implementations might provide predicted probabilities
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save the plot
        plt.savefig('roc_curve.png')
        print(f"ROC curve saved as 'roc_curve.png'")
    except:
        print("Could not generate ROC curve (probabilistic predictions not available)")
    
    return model


if __name__ == "__main__":
    train_and_evaluate()