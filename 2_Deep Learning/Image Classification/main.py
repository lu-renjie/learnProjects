import time
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import LeNet, NN
from logger import get_logger


log = True  # 如果不想将训练数据保存在logs文件夹中，就设为False
device = torch.device('mps')
root = '~/Documents/3_dataset'  # MNIST数据集的位置
batch_size = 32
lr = 0.01
epoch_num = 1


@torch.no_grad()
def evaluate(model, test_loader):
    correct_num, total_num = 0, 0
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        predictions = model(imgs).argmax(dim=1)
        correct_num += (predictions == labels).sum()
        total_num += len(imgs)
    return correct_num.item() / total_num


def main():
    train_loader, test_loader = get_dataloaders(root, batch_size)
    # model = NN().to(device)
    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epoch_num)

    logger = get_logger('logs', log=log)
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'learning_rate: {lr}')
    logger.info(f'epoch_num: {epoch_num}')

    start_time = time.time()
    step = 0
    for epoch in range(epoch_num):
        for imgs, labels in train_loader:
            step += 1

            optimizer.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)

            predictions = model(imgs)

            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if step % 50 == 0:
                logger.info(f'step: {step}, loss: {loss.item(): .4f}, accuracy: {evaluate(model, test_loader):.4f}')
        
        logger.info(f'epoch[{epoch + 1}/{epoch_num}], lr: {optimizer.param_groups[0]["lr"]: .4e}')
    logger.info(f'time consuming: {time.time() - start_time}')


if __name__ == '__main__':
    main()
