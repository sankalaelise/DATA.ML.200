class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # can be reused in the forward as it has no parameters.

        # muokattu 
        self.enc2=nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ) 
        # muokattu 
        self.enc3= nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        
        # Decoding path
        self.upconv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # muokattuaj
        self.upconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec3=nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size=3, padding=1),
                                nn.Sigmoid())


    def forward(self, x):
        """ 
        This function implements the forward pass
        
        Parameters:
            x: input images of shape (batch_size, 3, 96, 96)
        Returns: 
            denoised images of shape (batch_size, 3, 96, 96)
        """
        # YOUR CODE HERE
        down1 = self.enc1(x)
        down2 = self.pool(down1)
        down3 = self.enc2(down2)
        down4 = self.pool(down3)
        down5 = self.enc3(down4)

        up1 = self.upconv1(down5)
        concat1 = torch.cat([up1,down3], dim=1)
        up2 = self.dec1(concat1)
        up3 = self.upconv2(up2)
        concat2 = torch.cat([up3,down1], dim=1)
        up4 = self.dec2(concat2)
        output = self.dec3(up4)

        return output


def loss_and_optimizer(model):
    """
    This function initializes and returns the loss criterion and optimizer for training a CNN model.

    Parameters:
    model: The model to be trained.

    Returns:
    tuple: A tuple containing:
        -  The loss function (Mean Squared Error loss).
        -  The optimizer (Adam) for updating model parameters.
    """
    
    loss = nn.MSELoss(model)
    optimizer = optim.Adam(params=model.parameters(),lr = 0.001)

    return loss, optimizer


# Training loop

 #Training loop

def train(model, optimizer, criterion, train_loader, num_epochs=10, verbose=True):
    """
    Function to train the model.

    Parameters:
    - model: The model to train.
    - optimizer: The optimizer for updating model parameters.
    - criterion: The loss function to use.
    - train_loader: DataLoader for training data.
    - num_epochs: Number of training epochs (default is 10).
    - verbose: Boolean: print training progress and loss (default is True)
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = []                                  # Append the loss of every iteration to the epoch loss
        for i, (images, _) in enumerate(train_loader):

            images = images.to(device)                  # Move the input to 'device' (CPU or GPU)
            noisy_images = add_gaussian_noise(images)   # Add Gaussian noise to simulate noisy images

            # YOUR CODE HERE
            optimizer.zero_grad()
            y_pred = model(noisy_images)
            loss = criterion(y_pred,images)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(epoch_loss):.4f}')
            
# Testing loop
def test(model, criterion, test_loader):
    """ 
    Function to test the model.
    
    Parameters: 
    - model: The model to test.
    - criterion: The loss function to calculate test loss.
    - test_loader: DataLoader for testing data.
    """
    model.eval()
    test_loss = []
    # YOUR CODE HERE
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        noisy_images = add_gaussian_noise(images)
        y_pred = model(noisy_images)
        loss = criterion(y_pred,images)
        test_loss.append(loss.item())
        


    return np.mean(test_loss)