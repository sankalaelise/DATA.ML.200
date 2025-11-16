class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, non_linearity, apply_batchnorm, apply_dropout):
        super().__init__()
        
        self.apply_batchnorm = apply_batchnorm
        self.apply_dropout= apply_dropout
        
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride)

        if apply_batchnorm:
            self.bn = nn.BatchNorm1d(out_channels)
            
        if non_linearity == "ReLU":
            self.activation_fn = nn.ReLU()
        elif non_linearity == "Tanh":
            self.activation_fn = nn.Tanh()

        if apply_dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.apply_batchnorm:
            x = self.bn(x)     
        x = self.activation_fn(x)  
      
        if self.apply_dropout:
            x=self.dropout(x)
        return x


class MyModel(nn.Module):
    def __init__(self, nb_basic_blocks, conv_channels, kernel_size, stride, non_linearity, apply_batchnorm, apply_dropout, apply_pooling):
        super().__init__()
        self.apply_pooling = apply_pooling
        self.feature_extractor = None

        self.layers = nn.ModuleList()

        # your code here for initializing layers
        # ---------------------------------------------------------------------
        # In this task, you will design a flexible CNN model using the BasicBlock
        # defined above. Each BasicBlock may include:
        #   - Conv1d layer
        #   - Optional BatchNorm1d
        #   - Activation (ReLU or Tanh)
        #   - Optional Dropout
        #
        # Follow the steps below to build the model:
        #
        # 1. nn.ModuleList() is initialized and stored in `self.layers`
        # 2. Use a loop to create the specified number of BasicBlocks:
        #       - For the first block, set in_channels = 1
        #       - For the remaining blocks, set in_channels = conv_channels[i-1]
        #       - Set out_channels = conv_channels[i]
        #       - Pass kernel_size, stride, non_linearity, apply_batchnorm,
        #         and apply_dropout as parameters to BasicBlock.
        #       - Append each BasicBlock to `self.layers`.
        for i in range(nb_basic_blocks):
            if i == 0:
                in_channels = 1
            else: 
                in_channels = conv_channels[i-1]

            out_channels = conv_channels[i]
            new_block = BasicBlock(in_channels=in_channels,out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  non_linearity=non_linearity,
                                  apply_batchnorm=apply_batchnorm,
                                  apply_dropout=apply_dropout)
            self.layers.append(new_block)

            # 3. If apply_pooling = True:
            #       - After each BasicBlock in the loop iteration (except the last one),
            #         append a MaxPool1d layer to `self.layers`
            #         with the same kernel_size and stride.
            if apply_pooling and i < nb_basic_blocks -1:
                max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
                self.layers.append(max_pool)

        # 4. Once all layers are added, wrap them using nn.Sequential:
        self.feature_extractor = nn.Sequential(*self.layers)

        # 5. Add the remaining layers:
        #       - A Global Average Pooling layer (AdaptiveAvgPool1d)
        #       - A Flatten layer
        #       - If apply_pooling = True, also include a Linear (fc) layer
        #         that maps the last channel size to 1.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)     # näitten parametrit?
        self.flatten = nn.Flatten()

        
        if apply_pooling:
            self.fc = nn.Linear(conv_channels[-1],1)
        #else:
          #  self.linear = nn.Identity()

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # your code here for the forward pass
        # ---------------------------------------------------------------------
        # 1. Pass the input through the feature extractor (self.feature_extractor),
        #    which consists of multiple BasicBlocks and optional pooling layers.
        #
        # 2. Apply the global average pooling layer to reduce the spatial dimension.
        #
        # 3. Flatten the pooled output to a vector using the flatten layer.
        #
        # 4. If apply_pooling = True, pass the output through the fully connected layer.
        #
        # 5. Finally, apply the Sigmoid activation to obtain the final output.
        # ---------------------------------------------------------------------
        #
        # Hint:
        # The order should always be:
        # feature_extractor → global_avg_pool → flatten → (fc) → sigmoid

        # YOUR CODE HERE
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)

        #ensuring 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.apply_pooling:
            x = self.fc(x)        # size of x ->[batch,1]
        else: 
            if x.shape[1] == 1:
                x = x.view(-1, 1)
            else:
                x = x.mean(dim=1, keepdim=True)   # size to [batch,1] by mean

        x = self.sigmoid(x)
        return x


def get_num_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params} trainable parameters.')
    return num_params

def loss_and_optimizer(model, optimizer_type, learning_rate):

    # your code here
    # 1. Define the loss function as Binary Cross-Entropy Loss.
    # 2. Initialize the optimizer based on optimizer_type:
    #    - If 'Adam': use Adam optimizer.
    #    - If 'SGD': use SGD optimizer.
    # 3. Return both the criterion and optimizer.
    # YOUR CODE HERE
    loss = nn.BCELoss()

    optimizer = None
    if optimizer_type == "Adam":
        optimizer = optim.Adam(params=model.parameters(),lr =learning_rate)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(params=model.parameters(),lr =learning_rate)

    return loss, optimizer

base_config = {
    'nb_basic_blocks': 3,
    'conv_channels': [32,32,1],
    'kernel_size': 11,
    'stride': 5,
    'non_linearity': 'Tanh',
    'use_batchnorm': False,
    'use_dropout': False,
    'batch_size': 2,
    'shuffle': False,
    'optimizer_type': "Adam",
    'learning_rate': 0.0001,
    'apply_pooling': False
}


def training_loop(nb_epochs, model, optimizer, loss_fn, train_dataloader, test_dataloader, verbose=True):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    model.train()
    for epoch in range(1, nb_epochs + 1):
        start = time.time()
        train_loss, correct_predictions = 0., 0.
        num_samples = 0
        i = 0
        for i, (input_batch, target_batch) in enumerate(train_dataloader):
            # your code here for minibatch training
            # 1. call batch data and labels and set them to the correct device
            input_batch = input_batch.to(device)
            labels = target_batch.to(device).float().view(-1,1)
            # 2. make the prediction on the data
            predictions = model(input_batch)
            # 3. calculate loss
            loss_train = loss_fn(predictions, labels)
            # 4. set optimizer to zero grad
            optimizer.zero_grad()
            # 5. do backward pass
            loss_train.backward()
            # 6. move the optimizer one step forward
            optimizer.step()


            # accumulate correct prediction
            correct_predictions += ((predictions.detach() >= 0.5).int() == target_batch.int()).sum().item() # number of correct predictions
            train_loss += loss_train.item()
            num_samples += target_batch.size(0)

        average_train_loss = train_loss/(i+1)
        average_train_accuracy = correct_predictions/len(train_dataloader.dataset)

        test_loss, test_accuracy = testing_loop(model, loss_fn, test_dataloader)

        train_losses.append(average_train_loss)
        val_losses.append(test_loss)
        train_accuracies.append(average_train_accuracy)
        val_accuracies.append(test_accuracy)

        end = time.time()
        epoch_time = round(end - start, 2)
        if verbose:
            print(f'Epoch {epoch}, train_loss {average_train_loss:.2f}, train_accuracy: {average_train_accuracy:.4f},',
                  f'test_loss {test_loss:.2f}, test_accuracy: {test_accuracy:.4f}, time = {epoch_time}')

    return train_losses, val_losses, train_accuracies, val_accuracies
def testing_loop(model, loss_fn, test_dataloader):
    model.eval()
    with torch.no_grad():
        total_loss, correct_predictions = 0., 0.
        i = 0
        for i, (input_batch, target_batch) in enumerate(test_dataloader):
            # your code here for minibatch validation
            # 1. set input_batch, target_batch to correct device
            input_batch = input_batch.to(device)
            labels = target_batch.to(device).float().view(-1,1)
            # 2. make the prediction on input_batch
            predictions = model(input_batch)
            # 3. calculate loss and add it to previous loss
            loss_train = loss_fn(predictions, labels)
            
            total_loss += loss_train.item()
            # 4. obtain predicted class labels from predictions
            correct_predictions += ((predictions.detach() >= 0.5).int() == target_batch.int()).sum().item()

    # Average for all batches
    average_loss = total_loss / (i + 1)  # Use i + 1 for the total number of batches
    average_accuracy = correct_predictions / len(test_dataloader.dataset)

    return average_loss, average_accuracy