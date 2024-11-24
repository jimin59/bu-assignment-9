import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
import subprocess
import sys

# Ensure required packages are installed
try:
    import networkx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
import networkx as nx

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1  # Weights for input to hidden layer
        self.b1 = np.zeros((1, hidden_dim))  # Biases for hidden layer
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1  # Weights for hidden to output layer
        self.b2 = np.zeros((1, output_dim))  # Biases for output layer

    def forward(self, X):
        # Forward pass, apply layers to input X
        # Linear transformation for hidden layer
        Z1 = X.dot(self.W1) + self.b1
        
        # Activation for hidden layer
        if self.activation_fn == 'tanh':
            A1 = np.tanh(Z1)
        elif self.activation_fn == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        else:
            raise ValueError("Unsupported activation function")
        
        # Linear transformation for output layer
        Z2 = A1.dot(self.W2) + self.b2
        
        # Output activation (sigmoid for binary classification)
        out = 1 / (1 + np.exp(-Z2))
        
        # Store activations for visualization
        self.A1 = A1
        self.out = out
        
        return out

    def backward(self, X, y):
        # Compute gradients using chain rule
        m = X.shape[0]
        
        # Gradient of loss with respect to output (Cross-entropy loss derivative)
        dZ2 = self.out - y
        
        # Gradients for W2 and b2
        dW2 = (1 / m) * self.A1.T.dot(dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Gradient for hidden layer
        dA1 = dZ2.dot(self.W2.T)
        
        # Activation function derivative for hidden layer
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(self.A1) ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.A1 > 0)
        elif self.activation_fn == 'sigmoid':
            sigmoid_Z1 = 1 / (1 + np.exp(-self.A1))
            dZ1 = dA1 * sigmoid_Z1 * (1 - sigmoid_Z1)
        else:
            raise ValueError("Unsupported activation function")
        
        # Gradients for W1 and b1
        dW1 = (1 / m) * X.T.dot(dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        # Store gradients for visualization
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_xticks(np.arange(-1, 1.5, 0.5))
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_yticks(np.arange(-1, 1.5, 0.5))
    ax_hidden.set_zlim(-1, 1)
    ax_hidden.set_zticks(np.arange(-1, 1.5, 0.5))
    ax_hidden.set_title('Hidden Layer Features at Step {}'.format(frame * 10))
    ax_hidden.set_xlabel('Neuron 1')
    ax_hidden.set_ylabel('Neuron 2')
    ax_hidden.set_zlabel('Neuron 3')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, alpha=0.8, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap='bwr')
    ax_input.set_title('Input Space at Step {}'.format(frame * 10))

    # Visualize gradients as nodes and connections
    G = nx.DiGraph()
    positions = {}
    # Add nodes for input layer (x1, x2)
    for i in range(mlp.W1.shape[0]):
        G.add_node(f'x{i+1}', pos=(0, i))
        positions[f'x{i+1}'] = (0, i)
    # Add nodes for hidden layer (h1, h2, h3)
    for j in range(mlp.W1.shape[1]):
        G.add_node(f'h{j+1}', pos=(1, j))
        positions[f'h{j+1}'] = (1, j)
    # Add node for output layer (y)
    G.add_node('y', pos=(2, 0))
    positions['y'] = (2, 0)
    # Add edges with weights
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            G.add_edge(f'x{i+1}', f'h{j+1}', weight=mlp.dW1[i, j])
    for j in range(mlp.W2.shape[0]):
        G.add_edge(f'h{j+1}', 'y', weight=mlp.dW2[j, 0])
    # Draw nodes
    nx.draw_networkx_nodes(G, positions, ax=ax_gradient, node_color='blue', node_size=800, alpha=0.8)
    # Draw edges with varying thickness based on gradient magnitude
    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], ax=ax_gradient, width=np.abs(d['weight']) * 5, alpha=0.6, edge_color='purple')
    # Draw labels
    nx.draw_networkx_labels(G, positions, ax=ax_gradient, font_size=10, font_color='white')
    ax_gradient.set_title('Gradients at Step {}'.format(frame * 10))
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-1, max(positions.values(), key=lambda x: x[1])[1] + 1)
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
