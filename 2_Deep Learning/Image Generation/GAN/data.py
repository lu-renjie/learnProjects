from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_dataloaders(root, batch_size):
    transform = transforms.ToTensor()
    train_data = MNIST(root, train=True, transform=transform, download=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    return train_loader
