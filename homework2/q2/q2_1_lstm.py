import torch
import torch.nn as nn


class RNA_LSTM(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        # Bidirectional doubles the hidden dimension
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (Batch, 41, 4) -> No permutation needed for LSTM with batch_first=True
        
        # LSTM output: (Batch, Seq_Len, Hidden_Dim * 2)
        out, _ = self.lstm(x)
        
        out = torch.mean(out, dim=1) # Average over sequence length
        
        return self.fc(out)