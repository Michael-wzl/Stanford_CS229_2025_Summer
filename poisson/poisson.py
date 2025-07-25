import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    pr = PoissonRegression(step_size=lr)
    pr.fit(x_train, y_train)
    
    # Load validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    predictions = pr.predict(x_eval)
    np.savetxt(save_path, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_eval, predictions, alpha=0.6)
    plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--', lw=2)
    plt.xlabel('True Count')
    plt.ylabel('Predicted Expected Count')
    plt.title('True vs Predicted Counts (Validation Set)')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path.replace('.txt', '_plot.png'))
    plt.close()
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        
        for i in range(self.max_iter):
            eta = x.dot(self.theta)
            lambda_pred = np.exp(eta)
            gradient = x.T.dot(y - lambda_pred)
            theta_new = self.theta + self.step_size * gradient
            if np.linalg.norm(theta_new - self.theta) < self.eps:
                if self.verbose:
                    print(f'Converged after {i+1} iterations')
                break
            if self.verbose:
                print(f'Iteration {i}, Loss: {np.mean(y * eta - lambda_pred)}')
            self.theta = theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        eta = x.dot(self.theta)
        return np.exp(eta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
