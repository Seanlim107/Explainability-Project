import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from dataset_v2 import ASL_MNIST
from models import CNN_MNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            test_loss += criterion(output, targets).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
    return test_loss

def main():
    
    train_dataset = ASL_MNIST(csv_file=os.path.join('ASL_MNIST', 'train', 'sign_mnist.csv'))
    test_dataset = ASL_MNIST(csv_file=os.path.join('ASL_MNIST', 'test', 'sign_mnist.csv'))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CNN_MNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    min_test_loss = float('inf')
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss = test(model, device, test_loader, criterion)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join('checkpoints', 'asl_mnist_cnn.pth'))
            print(f'Model saved: Epoch {epoch+1}, Test loss: {test_loss:.4f}')

    print('Finished Training')

if __name__ == '__main__':
    main()
