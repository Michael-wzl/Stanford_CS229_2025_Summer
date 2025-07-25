import util
import numpy as np

def example_weights():
    """This is an example function that returns weights.
    Use this function as a template for optimal_step_weights and optimal_sigmoid_weights.
    You do not need to modify this function for this assignment.
    """
    w = {}

    w['hidden_layer_0_1'] = 0
    w['hidden_layer_1_1'] = 0
    w['hidden_layer_2_1'] = 0
    w['hidden_layer_0_2'] = 0
    w['hidden_layer_1_2'] = 0
    w['hidden_layer_2_2'] = 0
    w['hidden_layer_0_3'] = 0
    w['hidden_layer_1_3'] = 0
    w['hidden_layer_2_3'] = 0

    w['output_layer_0'] = 0
    w['output_layer_1'] = 0
    w['output_layer_2'] = 0
    w['output_layer_3'] = 0

    return w


def optimal_step_weights():
    """Return the optimal weights for the neural network with a step activation function.
    See the PDF for instructions on what each weight represents.
    The hidden layer weights are notated by [1] on the problem set and
    the output layer weights are notated by [2].
    This function should return a dict with elements for each weight, see example_weights above.
    """
    w = example_weights()

    # *** START CODE HERE ***
    # h1: detects x1 ≥ 0.698 (use -0.5 as approximation)
    w['hidden_layer_0_1'] = -0.5  # bias
    w['hidden_layer_1_1'] = 1     # x1 coefficient
    w['hidden_layer_2_1'] = 0     # x2 coefficient
    
    # h2: detects x2 ≥ 0.690 (use -0.5 as approximation)
    w['hidden_layer_0_2'] = -0.5  # bias
    w['hidden_layer_1_2'] = 0     # x1 coefficient
    w['hidden_layer_2_2'] = 1     # x2 coefficient
    
    # h3: detects x1 + x2 ≤ 4.0 (flip to: 4.0 - x1 - x2 ≥ 0)
    w['hidden_layer_0_3'] = 4     # bias (threshold = 4.0)
    w['hidden_layer_1_3'] = -1    # -x1 coefficient
    w['hidden_layer_2_3'] = -1    # -x2 coefficient
    
    # Output: Class 0 when h1=1 AND h2=1 AND h3=1 (inside region)
    # Class 1 otherwise
    # When all h1=h2=h3=1: output = 3-1-1-1 = 0 → Class 0
    # When any hi=0: output ≥ 1 → Class 1
    w['output_layer_0'] = 3       # bias
    w['output_layer_1'] = -1      # h1 coefficient
    w['output_layer_2'] = -1      # h2 coefficient
    w['output_layer_3'] = -1      # h3 coefficient
    # *** END CODE HERE ***

    return w

def optimal_linear_weights():
    """Return the optimal weights for the neural network with a linear activation function for the hidden units.
    See the PDF for instructions on what each weight represents.
    The hidden layer weights are notated by [1] on the problem set and
    the output layer weights are notated by [2].
    This function should return a dict with elements for each weight, see example_weights above.
    """
    w = example_weights()

    # *** START CODE HERE ***
    # h1: Main diagonal boundary (x1 + x2 - 4)
    # This captures the most important boundary: x1 + x2 = 4
    w['hidden_layer_0_1'] = -4.0  # bias (fine-tuned)
    w['hidden_layer_1_1'] = 1     # x1 coefficient
    w['hidden_layer_2_1'] = 1     # x2 coefficient
    
    # h2: Left boundary (x1 - 0.5)
    # Helps distinguish points on the left side
    w['hidden_layer_0_2'] = -0.5  # bias
    w['hidden_layer_1_2'] = 1     # x1 coefficient
    w['hidden_layer_2_2'] = 0     # x2 coefficient
    
    # h3: Bottom boundary (x2 - 0.5)
    # Helps distinguish points on the bottom side
    w['hidden_layer_0_3'] = -0.5  # bias
    w['hidden_layer_1_3'] = 0     # x1 coefficient
    w['hidden_layer_2_3'] = 1     # x2 coefficient
    
    # Output layer (step activation)
    # Combines all three boundaries with fine-tuned coefficients
    w['output_layer_0'] = -0.9    # bias (fine-tuned)
    w['output_layer_1'] = 1       # h1 coefficient (main diagonal)
    w['output_layer_2'] = 0.4     # h2 coefficient (left boundary, fine-tuned)
    w['output_layer_3'] = 0.55    # h3 coefficient (bottom boundary, fine-tuned)
    # *** END CODE HERE ***

    return w

def compute_predictions(X, weights, step_fn=True):
    def get(names):
        return [weights[name] for name in names]

    w11 = get(['hidden_layer_0_1', 'hidden_layer_1_1', 'hidden_layer_2_1'])
    w12 = get(['hidden_layer_0_2', 'hidden_layer_1_2', 'hidden_layer_2_2'])
    w13 = get(['hidden_layer_0_3', 'hidden_layer_1_3', 'hidden_layer_2_3'])
    wo = get(['output_layer_0', 'output_layer_1', 'output_layer_2', 'output_layer_3'])

    wo = np.array(wo)

    W1 = np.array([w11, w12, w13])

    if step_fn:
        h = (np.matmul(W1, X.transpose()) >= -1e-10)
    else:
        h = np.matmul(W1, X.transpose())

    H = np.concatenate((np.ones((1, X.shape[0])), h), axis=0)
    o = (np.matmul(H.transpose(), wo) >= 1e-10)

    return o

if __name__ == "__main__":
    x, y = util.load_dataset('train.csv', add_intercept=True)

    step_weights = optimal_step_weights()
    step_predict = lambda data: compute_predictions(data, step_weights, step_fn=True)
    step_y_pred = step_predict(x)
    print('Step function accuracy is: ' + str(np.mean(step_y_pred == y)))
    util.plot(x, y, step_predict, save_path='step_weights.pdf')

    linear_weights = optimal_linear_weights()
    linear_predict = lambda data: compute_predictions(data, linear_weights, step_fn=False)
    linear_y_pred = linear_predict(x)
    print('Linear function accuracy is: ' + str(np.mean(linear_y_pred == y)))
    util.plot(x, y, linear_predict, save_path='linear_weights.pdf')
