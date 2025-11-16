import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def generate_data_2d(num_samples=1000):
    # Generate random input data with shape (num_samples, 2)
    np.random.seed(4)
    x = np.random.randn(num_samples, 2)
    noise = np.random.randn(num_samples, 1) * 0.1  # Add small noise
    
    # Compute the output with the defined pattern
    y = np.sum(x**2, axis=1, keepdims=True) + noise

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    return x, y

def display_data(x,y):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), y[:,0].numpy(), c='b', marker='o')
    
    ax.set_xlabel('Feature 1 (x1)')
    ax.set_ylabel('Feature 2 (x2)')
    ax.set_zlabel('Target (y)', labelpad=0)
    ax.set_title('3D Plot of Synthetic Data with Two Features')
    
    plt.show()

x, y = generate_data_2d(1000)
display_data(x,y)
print(f'Shape of the input  x: {x.numpy().shape}')
print(f'Shape of the target y: {y.numpy().shape}')

class MLP(nn.Module):
    def __init__(self, in_features=2, hidden_size1=12, hidden_size2=10, hidden_size3=6, out_features=1):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_size1 = hidden_size1 
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.out_features = out_features

        # YOUR CODE HERE
        self.model = nn.Sequential(nn.Linear(in_features,hidden_size1),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size1,hidden_size2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size2,hidden_size3),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size3,out_features),)
    def forward(self, x):
        self.output = self.model.forward(x)
        return self.output
        
feedback_txt = []
# This cell checks the number of layers 
def test_layers():
    
    all_tests_successful = True
    model = MLP()
    relu_count, linear_count = 0, 0
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            relu_count += 1
        if isinstance(layer, nn.Linear):
            linear_count += 1
    
    if relu_count == 0:
        all_tests_successful = False
        msg = "At least one ReLU layer is expected, but got 0."
        feedback_txt.append(f"Visible test: {msg}")
        raise AssertionError(msg)

    if linear_count != 4:
        all_tests_successful = False
        msg = f"Expected 4 fully connected (Linear) layers, but got {linear_count}."
        feedback_txt.append(f"Visible test: {msg}")
        raise AssertionError(msg)

    if all_tests_successful:
        print("\033[92mVisible test for layer count passed successfully!\033[0m")
    
test_layers()

def init_model(learning_rate=0.05):
    """
    Initializes the model, loss function, and optimizer.
    
    Args:
    - learning_rate (float): Learning rate.
    
    Returns:
    - model (MLP): An instance of the MLP model.
    - criterion (nn.MSELoss): Mean squared error loss function for regression.
    - optimizer (torch.optim.Adam): Adam optimizer for updating model weights.
    
    Usage:
    Call this function before the training starts.
    """
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# Function to visualize the loss curve after training
def plot_loss_curve(losses, epochs):
    """
    Plots the training loss over epochs after the training loop.

    Args:
    - losses (list): List of loss values for each epoch.
    - epochs (int): Total number of epochs.

    Usage:
    After the training is completed, call this function to visualize how the training loss has changed over time.
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), losses, 'b', label='Training Loss')
    ax.grid(True)
    ax.set_title('Training Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.show()

def train_mlp(model, criterion, optimizer, x, y, epochs=200):
    losses = []
    for epoch in range(epochs):
        #YPUR CODE HERE
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        
    return losses  # Return to visualize the learning curve

# This cell tests the training code
from unittest.mock import patch
def test_train_calls():
    all_tests_successful = True
    model, criterion, optimizer = init_model(0.05)
    with patch('torch.Tensor.backward') as mock_backward, patch.object(optimizer, 'step') as mock_step:
        train_mlp(model, criterion, optimizer, x, y, epochs=10)
        if not mock_backward.called:
            all_tests_successful = False
            msg = "You forgot to calculate the gradients."
            feedback_txt.append(f"Visible test: {msg}")
            raise AssertionError(msg)
        if not mock_step.called:
            all_tests_successful = False
            msg = "Visible test: You forgot to update the weights."
            feedback_txt.append(f"Visible test: {msg}")
            raise AssertionError(msg)
        
        if all_tests_successful:
            print("\033[92mVisible test passed.\033[0m")
test_train_calls()

x, y = generate_data_2d(1000) # DO NOT OVERWRITE THIS
# Using recommended parameters to train the model (you can modify these as desired)
# You are not expected to implement anything
# To continue using the default parameters, remove raise NotImplementedError()
num_epochs = 200
learning_rate = 0.05
# Initialize the model, criterion, and optimizer
model, criterion, optimizer = init_model(learning_rate)
losses = train_mlp(model, criterion, optimizer, x, y, epochs=num_epochs)
plot_loss_curve(losses, len(losses))