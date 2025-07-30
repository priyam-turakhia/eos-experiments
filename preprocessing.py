import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

def flatten(X):
    return X.reshape(X.shape[0], -1)

def center(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    return X_train - mean, X_test - mean

def standardize(X_train, X_test):
    std = X_train.std(dim=0, keepdim=True)
    std[std == 0] = 1
    return X_train / std, X_test / std

def load_mnist(loss_fn: str):
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    X_train = flatten(train_data.data.float() / 255.0)
    X_test = flatten(test_data.data.float() / 255.0)
    y_train = train_data.targets
    y_test = test_data.targets
    
    X_train, X_test = center(X_train, X_test)
    X_train, X_test = standardize(X_train, X_test)
    
    X_train = X_train.view(-1, 1, 28, 28)
    X_test = X_test.view(-1, 1, 28, 28)
    
    if loss_fn == 'mse':
        y_train = F.one_hot(y_train, num_classes=10).float()
        y_test = F.one_hot(y_test, num_classes=10).float()
    
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)

def load_cifar10(loss_fn: str):
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    X_train = flatten(torch.tensor(train_data.data, dtype=torch.float32) / 255.0)
    X_test = flatten(torch.tensor(test_data.data, dtype=torch.float32) / 255.0)
    y_train = torch.tensor(train_data.targets)
    y_test = torch.tensor(test_data.targets)
    
    X_train, X_test = center(X_train, X_test)
    X_train, X_test = standardize(X_train, X_test)
    
    X_train = X_train.view(-1, 3, 32, 32)
    X_test = X_test.view(-1, 3, 32, 32)
    
    if loss_fn == 'mse':
        y_train = F.one_hot(y_train, num_classes=10).float()
        y_test = F.one_hot(y_test, num_classes=10).float()
    
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)

def load_cifar100(loss_fn: str):
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    X_train = flatten(torch.tensor(train_data.data, dtype=torch.float32) / 255.0)
    X_test = flatten(torch.tensor(test_data.data, dtype=torch.float32) / 255.0)
    y_train = torch.tensor(train_data.targets)
    y_test = torch.tensor(test_data.targets)
    
    X_train, X_test = center(X_train, X_test)
    X_train, X_test = standardize(X_train, X_test)
    
    X_train = X_train.view(-1, 3, 32, 32)
    X_test = X_test.view(-1, 3, 32, 32)
    
    if loss_fn == 'mse':
        y_train = F.one_hot(y_train, num_classes=100).float()
        y_test = F.one_hot(y_test, num_classes=100).float()
    
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)