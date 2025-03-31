from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_dataloaders(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    train_data = MNIST(root, train=True, transform=transform, download=False)
    test_data = MNIST(root, train=False, transform=transform, download=False)
    print(f'train set size: {len(train_data)}')
    print(f'test set size: {len(test_data)}')
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, shuffle=False)
    return train_loader, test_loader
