import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import models
import datasets


def collate_fn(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels)
    return texts, labels


def get_dataloader(dataset, batch_size, seed):
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader


@torch.no_grad()
def test(model, test_loader):
    model.eval()

    correct_num = 0
    for texts, labels in test_loader:
        logits = model(texts)
        predictions = logits.argmax(dim=1)

        labels = labels.to(logits.device)
        correct_num += (predictions == labels).sum()
    return correct_num.item()

def main(rank, args):
    # device = torch.device('cuda', rank)
    device = torch.device('cpu')
    world_size = len(args.gpu.split(','))
    dist.init_process_group('gloo', init_method=None, world_size=world_size, rank=rank)

    train_set = datasets.IMDB(train=True)
    train_loader = get_dataloader(train_set, args.batch_size, args.seed)
    test_set = datasets.IMDB(train=False)
    test_loader = get_dataloader(test_set, args.batch_size, args.seed)

    model = models.FastText(input_dim=50)
    model.to(device)
    model = DDP(model, find_unused_parameters=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    evaluate_interval = args.max_iteration // args.evaluate_times
    if rank == 0:
        print('evaluate interval', evaluate_interval)
    for step, (texts, labels) in enumerate(train_loader):
        if rank == 0:
            print(step)
        if step >= args.max_iteration:
            break

        model.train()
        labels = labels.to(device)

        logits = model(texts)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % evaluate_interval == 0:
            correct_num = test(model, test_loader)
            gather = [0, 0]
            dist.all_gather_object(gather, correct_num)
            if rank == 0:
                print('loss:', loss.item())
                print('accu:', sum(gather) / len(test_set))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--gpu', type=str, default='0', help='data parallism by "--gpu 0,1,2"')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size on each GPU')
    parser.add_argument('--max_iteration', type=int, default=1000, help='iteration num')
    parser.add_argument('--evaluate_times', type=int, default=50, help='evalute times')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    
    gpu_num = len(args.gpu.split(','))
    mp.spawn(main, args=(args,), nprocs=gpu_num, join=True)

