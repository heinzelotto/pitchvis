import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import random_split
from sklearn.metrics import f1_score
# Now you can use the data in PyTorch:
from torch.utils.data import TensorDataset, DataLoader

data = np.load('data.npy')

OCTAVES = 7
BUCKETS_PER_OCTAVE = 36
T = 5

def window_data(data, window_size):
    windows = []
    for i in range(data.shape[0] - window_size + 1):
        windows.append(data[i:i+window_size, :])
    return np.array(windows)

# Reshape the data to match the original structure
# data.shape[0] // (OCTAVES * BUCKETS_PER_OCTAVE + 128) gives us the number of data points
data = data.reshape((data.shape[0] // (OCTAVES * BUCKETS_PER_OCTAVE + 128), OCTAVES * BUCKETS_PER_OCTAVE + 128))

# Split the data into the cqt and midi arrays
cqt_data = data[:, :OCTAVES * BUCKETS_PER_OCTAVE]
midi_data = data[:, OCTAVES * BUCKETS_PER_OCTAVE:]

print (cqt_data.shape) # = (346616, 252)
cqt_data_windows = window_data(cqt_data, T)
print (cqt_data_windows.shape) # = (346612, 5, 252)
midi_data_windows = midi_data[T-1:]

# plot a grayscale spectrogram
# import matplotlib.pyplot as plt
# plt.imshow(cqt_data[22250:25000, :], cmap='gray_r', origin='lower')
# plt.show()

# Now, reshape for input into the model
cqt_data = cqt_data_windows.reshape(-1, 1, T, OCTAVES*BUCKETS_PER_OCTAVE)
print(cqt_data.shape)  # Output: (346612, 1, 5, 252)

# actually, just reshape for 1D convolution
cqt_data_windows = cqt_data_windows.reshape(-1, 1, T*OCTAVES*BUCKETS_PER_OCTAVE)

# Convert to PyTorch tensors
cqt_tensor = torch.from_numpy(cqt_data_windows).float()
midi_tensor = torch.from_numpy(midi_data_windows).float()

# Combine the tensors into a dataset
dataset = TensorDataset(cqt_tensor, midi_tensor)

# Determine the lengths of your splits, e.g., 80% training, 20% testing.
total_samples = len(dataset)
train_samples = int(total_samples * 0.8)
test_samples = total_samples - train_samples

# Randomly split the dataset.
train_dataset, test_dataset = random_split(dataset, [train_samples, test_samples])

# Create DataLoaders for each split.
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size, mlp_size, mlp_layers, output_size, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # TODO: maybe try 2d convolution again
        
        # Convolutional layer
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=0)
        O_conv = ((T*OCTAVES*BUCKETS_PER_OCTAVE - 5 + 2*0)/2) + 1
        O_pool = ((O_conv - 2 + 2*0)/2) + 1
        print(16*O_pool)
        self.fc1 = nn.Linear(16 * int(O_pool), mlp_size)  # This should be adapted to your input size
        
        for _ in range(mlp_layers):
            self.layers.append(nn.Linear(mlp_size, mlp_size))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(mlp_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply Convolutional layer
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return self.sigmoid(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Training parameters
mlp_layers = 2
mlp_size = 1024
mlp_dropout = 0.1
input_size = 252  # The input dimensions
output_size = 128  # The output dimensions

model = MLP(input_size, mlp_size, mlp_layers, output_size, mlp_dropout)
print ("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
# model_epochs = 32
# model_batch_size = 100
# model_workers = 64
# adam_learning_rate = 0.00001
# adam_weight_decay = 0.0005
# adam_beta1 = 0.9
# adam_beta2 = 0.999
# adam_epsilon = 1.1920929e-7

# Hyperparameters
model_epochs = 32
model_batch_size = 300
model_workers = 64
adam_learning_rate = 0.00001
adam_weight_decay = 0.0005
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1.1920929e-7

# Set up optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=adam_learning_rate, 
                       betas=(adam_beta1, adam_beta2), 
                       eps=adam_epsilon, 
                       weight_decay=adam_weight_decay)
criterion = nn.BCELoss()  # Since you have mentioned intensities in the range [0, 1], Binary Cross Entropy Loss is chosen.

# Training loop
for epoch in range(model_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

print('Finished Training, starting testing')

model.eval()  # Set the model to evaluation mode
f1_scores = []  # To store F1 scores of each batch
total = 0
correct = 0
with torch.no_grad():  # Disable gradient computation
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)

        # The outputs are probabilities, to convert to predicted class we choose the one with highest probability
        predicted = outputs.data > 0.5
        labels = labels > 0.5

        # Convert tensors to NumPy arrays
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

        # Calculate and store F1 score for this batch
        f1 = f1_score(labels, predicted, average='micro')  # 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        f1_scores.append(f1)

        # Calculate accuracy for this batch
        total += labels.size
        correct += (predicted == labels).sum().item()

# Calculate average F1 score across all batches
average_f1_score = sum(f1_scores) / len(f1_scores)

print('Average F1 Score on the test data: %.2f' % average_f1_score)

# Calculate accuracy across all batches
print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

# Here we're using torch.jit.trace to generate a TorchScript module
# You can also use torch.jit.script for models with control flow (e.g. if/while statements)
# input_example is a tensor that has the same dimension as your model input.
input_example = torch.randn(1, 1, T * OCTAVES * BUCKETS_PER_OCTAVE).to(device) # Batch size 1, 1 channel, 252 dimensions

traced_script_module = torch.jit.trace(model, input_example)

# Save the TorchScript module
torch.jit.save(traced_script_module, "model.pt")