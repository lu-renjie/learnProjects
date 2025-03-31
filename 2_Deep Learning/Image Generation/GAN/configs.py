import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='~/Documents/dataset')
parser.add_argument('--epoch_num', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--noise_dim', type=int, default=100)

parser.add_argument('--log', required=True, choices=['true', 'false'], type=str)
parser.add_argument('--lr_G', required=True, type=float)
parser.add_argument('--lr_D', required=True, type=float)

args = parser.parse_args()
device = torch.device('cpu')
