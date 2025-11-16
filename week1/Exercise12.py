import matplotlib.pyplot as plt
import numpy as np
from IPython import display

class Linear:
    def __init__(self, input_dim, output_dim, initial_weights=None, initial_biases=None):
        if initial_weights is None: initial_weights = np.random.randn(output_dim, input_dim)
        if initial_biases is None: initial_biases = np.random.randn(output_dim)
        self.weights = initial_weights
        self.biases = initial_biases
        
    def forward(self, x):
        self.x = x 
        self.output = np.dot(x,np.transpose(self.weights)) + self.biases                            # muokattu
        print(self.output)
        return self.output
    
    def backward(self, grad_output):
        assert hasattr(self, 'x'), 'Perform forward pass first.'
        self.grad_weights = np.dot(np.transpose(grad_output),self.x)                                # muokattu            
        self.grad_biases = sum(grad_output)                                                         # muokattu
        self.grad_input  = np.dot(grad_output,self.weights)                                         # muokattu

        return self.grad_input
    
    def update_params(self, learning_rate):
        self.weights = self.weights-learning_rate*self.grad_weights 
        self.biases = self.biases-learning_rate*self.biases

class Tanh:
    def forward(self, x):
        self.x = x 
        self.output = np.tanh(x)                            # muokattu
        return self.output
    
    def backward(self, grad_output):
        assert hasattr(self, 'x'), 'Perform forward pass first.'

        z = np.tanh(self.x)
        self.grad_input = 1-z**2
        return self.grad_input

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.activation = Tanh()
        self.linear2 = Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        hidden = self.linear1.forward(x)
        activated_hidden = self.activation.forward(hidden) 
        output = self.linear2.forward(activated_hidden)
        return output
    
    def backward(self, grad_output):
        dLdz2 = self.linear2.backward(grad_output)
        dLdz1 = self.activation.backward(dLdz2)
        grad_input = self.linear1.backward(dLdz1)
        
        return grad_input
    
    def update_params(self, learning_rate):
        self.linear1.update_params(learning_rate)
        self.linear2.update_params(learning_rate)

class MSELoss:
    def forward(self, y, t):
        self.y = y
        self.t = t
        
        loss = np.mean((y-t)**2)
        return loss
    
    def backward(self):
        grad_input = 2/np.size(self.y)*(self.y-self.t)
        return grad_input

def test_linear():
    init_weights = np.array([[1.0,0.,-1.0],[2.0,-1.0,0.0]])
    init_bias = np.array([0.5,-0.5])
    linear = Linear(input_dim=3, output_dim=2,initial_weights=init_weights,initial_biases=init_bias)

    # Syöte (batch_size=2)
    x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

    # Simuloidaan gradientti ulostulosta (esim. dL/dy)
    grad_output = np.array([[1.0, -1.0],
                        [0.5, 0.5]])

    # --- Testaa forward ---
    y = linear.forward(x)
    #print("FORWARD output:\n", y)

    # --- Testaa backward ---
    grad_input = linear.backward(grad_output)
    """
    print("\nGRADIENTS:")
    print("grad_weights:\n", linear.grad_weights)
    print("grad_biases:\n", linear.grad_biases)
    print("grad_input (dL/dx):\n", grad_input)
"""
    # --- Tallennetaan painot ennen päivitystä ---
    old_weights = linear.weights.copy()
    old_biases = linear.biases.copy()

    # --- Testaa päivitys ---
    lr = 0.01
    linear.update_params(lr)

    print("\nUPDATED PARAMETERS:")
    print("New weights:\n", linear.weights)
    print("ΔW (muutos):\n", linear.weights - old_weights)
    print("New biases:\n", linear.biases)
    print("Δb (muutos):\n", linear.biases - old_biases)

test_linear()


def test_loss():
    y = np.array([[0.2], [0.8], [0.5]])
    t = np.array([[0.0], [1.0], [0.0]])

    loss_fn = MSELoss()
    loss = loss_fn.forward(y,t)
    grad = loss_fn.backward()

    print("Loss:", loss)
    print("Grad shape:", grad.shape)
    print("Grad values:\n", grad)

test_loss()


