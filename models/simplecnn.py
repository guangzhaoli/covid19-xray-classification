import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Assuming input size is 64x64, after two pooling layers:
        # Input -> Conv1 -> Pool -> Conv2 -> Pool
        # Size reduces from 64x64 -> 32x32 -> 16x16 -> 32*16*16 features for fc1
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Adjust based on input dimensions
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, feature_size)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
