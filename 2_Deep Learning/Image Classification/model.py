import torch
import torch.nn as nn


def weights_init(m):
    """
    LeNet初始化方式改成这个收敛会快很多
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.apply(weights_init)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 28 * 28
        x = self.maxpool(x)  # 14, 14
        x = torch.relu(self.conv2(x))  # 10 * 10
        x = self.maxpool(x)  # 5 * 5
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

        def init_zero(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, val=0.0)
                torch.nn.init.constant_(m.bias, val=0.0)
        self.apply(init_zero)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)
