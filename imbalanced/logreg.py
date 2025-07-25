import numpy as np

class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True, l2reg=0.):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        self.l2reg = l2reg
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        
        # Initialize theta if not provided
        if self.theta is None:
            self.theta = np.zeros(d)
        
        for i in range(self.max_iter):
            # Forward pass: compute predictions
            h = self._sigmoid(x.dot(self.theta))

            eps_stable = 1e-15
            # Compute loss for numerical stability (add small epsilon)
            eps_stable = 1e-15
            h = np.clip(h, eps_stable, 1 - eps_stable)
            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + self.l2reg * np.sum(self.theta[1:] ** 2) / 2

            # Compute gradient
            gradient = (1/n) * x.T.dot(h - y)
            gradient[1:] += self.l2reg * self.theta[1:]

            # Check for convergence
            if np.linalg.norm(gradient) < self.eps:
                if self.verbose:
                    print(f'Converged after {i+1} iterations')
                break
                
            # Update parameters
            self.theta -= self.learning_rate * gradient
            
            # Print loss occasionally
            if self.verbose and i % 1000 == 0:
                print(f'Iteration {i}, Loss: {loss:.6f}')
        # *** END CODE HERE ***
    
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self._sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***
