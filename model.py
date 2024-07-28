import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """A simple fully connected network suitable for regression tasks."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.MSELoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for features, targets in trainloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = net(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set and report loss."""
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            features, targets = data[0].to(device), data[1].to(device)
            predictions = net(features)
            loss = criterion(predictions, targets).item()
            total_loss += loss
    average_loss = total_loss / len(testloader)
    return average_loss
