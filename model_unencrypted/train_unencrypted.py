import torch
import torch.nn as nn
import torchvision

from train_utils import ResNet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# model  = ResNet_orig()
# inp = torch.randn(5, 3, 32, 32)
# model(inp)
model = ResNet(4, out_chan=8)
inp = torch.randn(3, 1, 24, 24)
output = model(inp)