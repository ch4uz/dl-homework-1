from torch import nn

class Q1Net(nn.Module):
    def __init__(self, no_softmax=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=128 * 28 * 28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=8)
        ]
        if no_softmax == False:
            layers.append(nn.Softmax(dim=1))
        self.stack = nn.Sequential(*layers)


    def forward(self, x):
        logits = self.stack(x)
        return logits
