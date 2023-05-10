import torch
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = torch.nn.Linear(256, 128)
        self.relu1 = torch.nn.ReLU()
        self.drp1 = nn.Dropout(p=0.1)

        self.fc2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.drp2 = nn.Dropout(p=0.1)

        self.fc3 = torch.nn.Linear(64, 32)
        self.relu3 = torch.nn.ReLU()
        self.drp3 = nn.Dropout(p=0.1)

        self.fc4 = torch.nn.Linear(32, 1)
        self.drp4 = nn.Dropout(p=0.1)
        self.flat = nn.Flatten()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drp1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drp2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drp3(out)

        out = self.fc4(out)
        return out