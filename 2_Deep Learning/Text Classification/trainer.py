"""
TODO: 
* 加上verbose, 告知是否用多进程、batch数量、数据集信息等
* 分布式测试要改，把分布式的代码剥离
* 分布式训练和单进程训练是分开写还是一起写？
* 加上模型并行
* 加上进度条
"""

import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import models


def collate_fn(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels)
    return texts, labels


class Trainer:
    def __init__(self):
        pass

    def set_arguments(self):
        parser = ArgumentParser()
        parser.add_argument('--gpu', type=str, default='0', help='data parallism by "--gpu 0,1,2"')
        parser.add_argument('--seed', type=int, default=0, help='random seed')

        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size on each GPU')
        parser.add_argument('--max_iteration', type=int, default=1000, help='iteration num')
        parser.add_argument('--evaluate_times', type=int, default=50, help='evalute times')
        args = parser.parse_args()
        return args

    def set_train_set(self, args):
        train_set = datasets.IMDB(train=True)
        return train_set

    def set_test_set(self, args):
        test_set = datasets.IMDB(train=False)
        return test_set

    def set_model(self, args):
        model = models.FastText(input_dim=50)
        return model

    def set_optimizer(self, args, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return optimizer
    
    def set_loss_fn(self, args):
        return torch.nn.CrossEntropyLoss()

    def train_step(self, args, model, batch, device):
        texts, labels = batch
        labels = labels.to(device)
        loss_fn = self.set_loss_fn(args)
        logits = model(texts)
        loss = loss_fn(logits, labels)
        return loss

    def log(self, *args):
        """
        TODO: 加上时间前缀、写入文件
        """
        if self.distributed:
            rank = dist.get_rank()
            if rank == 0:
                print(*args)
        else:
            print(*args)

    def evaluate(self, args, model):
        model.eval()

        test_set = self.set_test_set(args)
        test_sampler, test_loader = self._get_dataloader(test_set, args.batch_size, args.seed)

        correct_num = 0
        for texts, labels in test_loader:
            logits = model(texts)
            predictions = logits.argmax(dim=1)

            labels = labels.to(logits.device)
            correct_num += (predictions == labels).sum()

        if self.distributed:
            gather = [0, 0]
            dist.all_gather_object(gather, correct_num)
            self.log('accu:', sum(gather) / len(test_set))
        else:
            self.log('accu:', correct_num / len(test_set))

    def _get_dataloader(self, dataset, batch_size, seed):
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, collate_fn=collate_fn)
        else:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, batch_size, sampler=sampler, collate_fn=collate_fn)
        return sampler, dataloader

    def _train(self, rank, args):
        if self.distributed:
            world_size = len(args.gpu.split(','))
            dist.init_process_group('gloo', init_method=None, world_size=world_size, rank=rank)
            self.device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # data
        train_set = self.set_train_set(args)
        train_sampler, train_loader = self._get_dataloader(train_set, args.batch_size, args.seed)

        # model
        model = self.set_model(args).to(self.device)

        if self.distributed:
            model = DDP(model, find_unused_parameters=False)
            # TODO: 优化掉这个find_unsed_parameters

        # optimizer
        optimizer = self.set_optimizer(args, model)
        
        # train
        evaluate_interval = args.max_iteration // args.evaluate_times
        iteration_of_epoch = len(train_loader)
        epoch_num = args.max_iteration / iteration_of_epoch
        self.log('evaluate interval', evaluate_interval)
        self.log('iteration_of_epoch', iteration_of_epoch)
        self.log('epoch_num', epoch_num)

        iteration = 0
        for epoch in range(math.ceil(epoch_num)):
            train_sampler.set_epoch(epoch)
            if self.distributed:
                train_sampler.set_epoch(epoch)

            for i, batch in enumerate(train_loader):
                model.train()
                loss = self.train_step(args, model, batch, self.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1
                self.log('iteration:', iteration, 'i', i, 'epoch:', epoch)
                if iteration >= args.max_iteration:
                    break

                # test
                if iteration % evaluate_interval == 0:
                    with torch.no_grad():
                        self.evaluate(args, model)
        self.log('finished training')    

    def _run(self):
        args = self.set_arguments()

        gpu_num = len(args.gpu.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if gpu_num > 1:
            print('use distributed training')
            self.distributed = True
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '1234'  # TODO: 自动查找端口可以吗
            mp.spawn(self._train, args=(args,), nprocs=gpu_num, join=True)
        else:
            self.distributed = False
            self._train(rank=None, args=args)
        

if __name__ == '__main__':
    Trainer()._run()
