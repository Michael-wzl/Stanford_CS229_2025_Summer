import numpy as np
import util
import sys
from random import random

#sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_vanilla = save_path.replace(WILDCARD, 'vanilla')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path)
    x_val, y_val = util.load_dataset(validation_path)
    class_0_mask = (y_val == 0)
    class_1_mask = (y_val == 1)
    
    # Part (b): Vanilla logistic regression
    print("Training vanilla logistic regression...")
    lgv = LogisticRegression()
    lgv.fit(x_train, y_train)

    y_pred_vanilla = lgv.predict(x_val)

    np.savetxt(output_path_vanilla, y_pred_vanilla)

    y_pred_vanilla = (y_pred_vanilla >= 0.5).astype(int)
    accuracy_vanilla = np.mean(y_pred_vanilla == y_val)
    accuracy_0_vanilla = np.mean(y_pred_vanilla[class_0_mask] == y_val[class_0_mask]) if np.sum(class_0_mask) > 0 else 0
    accuracy_1_vanilla = np.mean(y_pred_vanilla[class_1_mask] == y_val[class_1_mask]) if np.sum(class_1_mask) > 0 else 0
    balanced_accuracy_vanilla = (accuracy_0_vanilla + accuracy_1_vanilla) / 2

    print(f"Vanilla Logistic Regression Results:")
    print(f"Overall Accuracy: {accuracy_vanilla:.4f}")
    print(f"balanced Accuracy: {balanced_accuracy_vanilla:.4f}")
    print(f"Class 0 Accuracy: {accuracy_0_vanilla:.4f}")
    print(f"Class 1 Accuracy: {accuracy_1_vanilla:.4f}")
    
    util.plot(x_val, y_val, lgv.theta, output_path_vanilla.replace('.txt', '.png'))
    
    # Part (d): Upsampling minority class
    print("\nTraining logistic regression with upsampling...")
    
    minority_mask = (y_train == 1)
    majority_mask = (y_train == 0)
    
    x_minority = x_train[minority_mask]
    y_minority = y_train[minority_mask]
    x_majority = x_train[majority_mask]
    y_majority = y_train[majority_mask]
    
    repeat_times = int(1 / kappa)
    x_minority_upsampled = np.tile(x_minority, (repeat_times, 1))
    y_minority_upsampled = np.tile(y_minority, repeat_times)

    x_train_upsampled = np.vstack([x_majority, x_minority_upsampled])
    y_train_upsampled = np.hstack([y_majority, y_minority_upsampled])
    
    lgu = LogisticRegression()
    lgu.fit(x_train_upsampled, y_train_upsampled)
    
    y_pred_upsampling = lgu.predict(x_val)

    np.savetxt(output_path_upsampling, y_pred_upsampling)
    
    y_pred_upsampling = (y_pred_upsampling >= 0.5).astype(int)
    accuracy_upsampling = np.mean(y_pred_upsampling == y_val)
    accuracy_0_upsampling = np.mean(y_pred_upsampling[class_0_mask] == y_val[class_0_mask]) if np.sum(class_0_mask) > 0 else 0
    accuracy_1_upsampling = np.mean(y_pred_upsampling[class_1_mask] == y_val[class_1_mask]) if np.sum(class_1_mask) > 0 else 0
    balanced_accuracy_upsampling = (accuracy_0_upsampling + accuracy_1_upsampling) / 2
    
    print(f"Upsampling Logistic Regression Results:")
    print(f"Overall Accuracy (A): {accuracy_upsampling:.4f}")
    print(f"Balanced Accuracy (A_bar): {balanced_accuracy_upsampling:.4f}")
    print(f"Class 0 Accuracy (A0): {accuracy_0_upsampling:.4f}")
    print(f"Class 1 Accuracy (A1): {accuracy_1_upsampling:.4f}")
    
    util.plot(x_val, y_val, lgu.theta, 'imbalanced_upsampling_pred.png')
    
    print(f"\nComparison:")
    print(f"Improvement in Class 1 Accuracy: {accuracy_1_upsampling - accuracy_1_vanilla:.4f}")
    print(f"Change in Class 0 Accuracy: {accuracy_0_upsampling - accuracy_0_vanilla:.4f}")
    print(f"Improvement in Balanced Accuracy: {balanced_accuracy_upsampling - balanced_accuracy_vanilla:.4f}")
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
