#!/usr/bin/env python3
"""
Test script for evaluating trained neural network models on test data.
This script loads the saved parameters and evaluates both baseline and regularized models.
"""

import numpy as np
from nn import *

def load_and_test_model(model_name, all_data, all_labels):
    """Load saved parameters and test the model"""
    # Load saved parameters
    params = np.load(f'./{model_name}_params.npy', allow_pickle=True).item()
    
    # Test on test set
    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print(f'{model_name.capitalize()} model test accuracy: {accuracy:.6f}')
    
    return accuracy

def main():
    print("Loading test data...")
    
    # Load and preprocess data (same as training)
    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    # Normalize using training set statistics
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    print("\nTesting models on test set:")
    print("=" * 50)
    
    # Test baseline model
    baseline_acc = load_and_test_model('baseline', all_data, all_labels)
    
    # Test regularized model
    regularized_acc = load_and_test_model('regularized', all_data, all_labels)
    
    print("\nComparison:")
    print(f"Baseline model:    {baseline_acc:.6f}")
    print(f"Regularized model: {regularized_acc:.6f}")

if __name__ == '__main__':
    main()
