import numpy as np
import matplotlib.pyplot as plt
import util

class LogisticRegressionAnalysis:
    """Logistic regression for analyzing theta norm convergence."""
    
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.theta_history = []
        self.norm_history = []
        self.loss_history = []
        self.iteration_history = []
        
    def fit(self, x, y):
        """Run gradient descent and record theta norm at each iteration."""
        n, d = x.shape
        self.theta = np.zeros(d)
        
        # Clear history
        self.theta_history = []
        self.norm_history = []
        self.loss_history = []
        self.iteration_history = []
        
        for i in range(self.max_iter):
            # Forward pass: compute predictions
            h = self._sigmoid(x.dot(self.theta))
            
            # Compute loss for numerical stability
            eps_stable = 1e-15
            h = np.clip(h, eps_stable, 1 - eps_stable)
            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            
            # Compute gradient
            gradient = (1/n) * x.T.dot(h - y)
            
            # Record history every 1000 iterations or at the end
            if i % 1000 == 0 or i == self.max_iter - 1:
                self.theta_history.append(self.theta.copy())
                self.norm_history.append(np.linalg.norm(self.theta))
                self.loss_history.append(loss)
                self.iteration_history.append(i)
                
                if self.verbose:
                    print(f'Iteration {i}, Loss: {loss:.6f}, ||theta||: {np.linalg.norm(self.theta):.6f}')
            
            # Check for convergence
            if np.linalg.norm(gradient) < self.eps:
                # Record final state
                self.theta_history.append(self.theta.copy())
                self.norm_history.append(np.linalg.norm(self.theta))
                self.loss_history.append(loss)
                self.iteration_history.append(i)
                print(f'Converged after {i+1} iterations, Final ||theta||: {np.linalg.norm(self.theta):.6f}')
                break
                
            # Update parameters
            self.theta -= self.learning_rate * gradient
    
    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


def analyze_datasets():
    """analyze the theta norm behavior during training on both datasets."""

    # Load datasets
    x_train_a, y_train_a = util.load_csv('ds1_a.csv', add_intercept=True)
    x_train_b, y_train_b = util.load_csv('ds1_b.csv', add_intercept=True)
    
    # Test different max_iter values
    max_iter_values = [1000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    results_a = {}
    results_b = {}
    
    print("=== Analyzing Dataset A ===")
    for max_iter in max_iter_values:
        print(f"\n--- Max iterations: {max_iter} ---")
        clf_a = LogisticRegressionAnalysis(max_iter=max_iter, verbose=True)
        clf_a.fit(x_train_a, y_train_a)
        
        final_norm = clf_a.norm_history[-1] if clf_a.norm_history else 0
        final_loss = clf_a.loss_history[-1] if clf_a.loss_history else float('inf')
        
        results_a[max_iter] = {
            'final_norm': final_norm,
            'final_loss': final_loss,
            'norm_history': clf_a.norm_history,
            'iteration_history': clf_a.iteration_history,
            'final_theta': clf_a.theta.copy()
        }
    
    print("\n=== Analyzing Dataset B ===")
    for max_iter in max_iter_values:
        print(f"\n--- Max iterations: {max_iter} ---")
        clf_b = LogisticRegressionAnalysis(max_iter=max_iter, verbose=True)
        clf_b.fit(x_train_b, y_train_b)
        
        final_norm = clf_b.norm_history[-1] if clf_b.norm_history else 0
        final_loss = clf_b.loss_history[-1] if clf_b.loss_history else float('inf')
        
        results_b[max_iter] = {
            'final_norm': final_norm,
            'final_loss': final_loss,
            'norm_history': clf_b.norm_history,
            'iteration_history': clf_b.iteration_history,
            'final_theta': clf_b.theta.copy()
        }
    
    # Plot results
    plot_theta_norm_analysis(results_a, results_b, max_iter_values)
    
    return results_a, results_b


def plot_theta_norm_analysis(results_a, results_b, max_iter_values):
    """plot the theta norm analysis results."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Final theta norm vs max_iter
    final_norms_a = [results_a[max_iter]['final_norm'] for max_iter in max_iter_values]
    final_norms_b = [results_b[max_iter]['final_norm'] for max_iter in max_iter_values]
    
    ax1.plot(max_iter_values, final_norms_a, 'b-o', label='Dataset A', linewidth=2, markersize=6)
    ax1.plot(max_iter_values, final_norms_b, 'r-s', label='Dataset B', linewidth=2, markersize=6)
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Final ||θ||')
    ax1.set_title('Final θ Norm vs Max Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Final loss vs max_iter
    final_losses_a = [results_a[max_iter]['final_loss'] for max_iter in max_iter_values]
    final_losses_b = [results_b[max_iter]['final_loss'] for max_iter in max_iter_values]
    
    ax2.plot(max_iter_values, final_losses_a, 'b-o', label='Dataset A', linewidth=2, markersize=6)
    ax2.plot(max_iter_values, final_losses_b, 'r-s', label='Dataset B', linewidth=2, markersize=6)
    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss vs Max Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: Theta norm evolution for Dataset A (multiple max_iter)
    colors_a = plt.cm.Blues(np.linspace(0.3, 1, len([50000, 100000, 200000])))
    for i, max_iter in enumerate([50000, 100000, 200000]):
        if max_iter in results_a:
            ax3.plot(results_a[max_iter]['iteration_history'], 
                    results_a[max_iter]['norm_history'], 
                    color=colors_a[i], linewidth=2, 
                    label=f'Max iter: {max_iter}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('||θ||')
    ax3.set_title('θ Norm Evolution - Dataset A')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Theta norm evolution for Dataset B (multiple max_iter)
    colors_b = plt.cm.Reds(np.linspace(0.3, 1, len([50000, 100000, 200000])))
    for i, max_iter in enumerate([50000, 100000, 200000]):
        if max_iter in results_b:
            ax4.plot(results_b[max_iter]['iteration_history'], 
                    results_b[max_iter]['norm_history'], 
                    color=colors_b[i], linewidth=2, 
                    label=f'Max iter: {max_iter}')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('||θ||')
    ax4.set_title('θ Norm Evolution - Dataset B')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('theta_norm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_decision_boundaries():
    """plot decision boundaries at different iteration counts"""
    x_train_a, y_train_a = util.load_csv('ds1_a.csv', add_intercept=True)
    x_train_b, y_train_b = util.load_csv('ds1_b.csv', add_intercept=True)
    
    max_iters = [10000, 50000, 200000]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Dataset A
    for i, max_iter in enumerate(max_iters):
        clf = LogisticRegressionAnalysis(max_iter=max_iter)
        clf.fit(x_train_a, y_train_a)
        
        ax = axes[0, i]
        # Plot data points
        ax.plot(x_train_a[y_train_a == 1, 1], x_train_a[y_train_a == 1, 2], 'bx', linewidth=2, markersize=8, label='Class 1')
        ax.plot(x_train_a[y_train_a == 0, 1], x_train_a[y_train_a == 0, 2], 'go', linewidth=2, markersize=8, label='Class 0')
        
        # Plot decision boundary
        if len(clf.theta) >= 3:
            x1_range = np.linspace(x_train_a[:, 1].min(), x_train_a[:, 1].max(), 100)
            x2_boundary = -(clf.theta[0] + clf.theta[1] * x1_range) / clf.theta[2]
            ax.plot(x1_range, x2_boundary, 'r-', linewidth=3, label='Decision Boundary')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Dataset A - Max Iter: {max_iter}\n||θ|| = {np.linalg.norm(clf.theta):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Dataset B
    for i, max_iter in enumerate(max_iters):
        clf = LogisticRegressionAnalysis(max_iter=max_iter)
        clf.fit(x_train_b, y_train_b)
        
        ax = axes[1, i]
        # Plot data points
        ax.plot(x_train_b[y_train_b == 1, 1], x_train_b[y_train_b == 1, 2], 'bx', linewidth=2, markersize=8, label='Class 1')
        ax.plot(x_train_b[y_train_b == 0, 1], x_train_b[y_train_b == 0, 2], 'go', linewidth=2, markersize=8, label='Class 0')
        
        # Plot decision boundary
        if len(clf.theta) >= 3:
            x1_range = np.linspace(x_train_b[:, 1].min(), x_train_b[:, 1].max(), 100)
            x2_boundary = -(clf.theta[0] + clf.theta[1] * x1_range) / clf.theta[2]
            ax.plot(x1_range, x2_boundary, 'r-', linewidth=3, label='Decision Boundary')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Dataset B - Max Iter: {max_iter}\n||θ|| = {np.linalg.norm(clf.theta):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_boundaries_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    results_a, results_b = analyze_datasets()
    plot_decision_boundaries()
