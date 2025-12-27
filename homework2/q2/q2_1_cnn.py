import torch.nn as nn

class RNA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (Batch, 41, 4) -> Transpose to (Batch, 4, 41) for Conv1d
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) 
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        
        self.global_pool = nn.AdaptiveMaxPool1d(1) # Flatten to (Batch, 64, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Regression output
        )

    def forward(self, x):
        # x shape: (Batch, 41, 4)
        x = x.permute(0, 2, 1) # Change to (Batch, Channels, Length) for PyTorch CNN
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1) # Flatten
        
        return self.fc(x)