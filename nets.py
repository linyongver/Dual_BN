import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1_weight=Parameter(torch.Tensor(hidden_size,input_size))
        self.ll_bias=Parameter(torch.Tensor(hidden_size))
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(num_features=hidden_size)
        self.fc1 =nn.Linear(input_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,num_classes)

    def first_layer_precess(self,x):
        self.out_l1=self.fc1(x)
        self.out_rl1=self.relu(self.out_l1)

    def forward(self,x):
        self.first_layer_precess(x)
        self.out_last= self.fc2(self.out_rl1)
        return self.out_last
    
class NeuralNetTorchNorm(NeuralNet):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetTorchNorm,self).__init__(input_size,hidden_size,num_classes)
        print("nets::NeuralNetTorchNorm,using BN from Torch.")
    
    def first_layer_precess(self,x):
        self.out_l1=self.fc1(x)
        self.out_rll=self.relu(self.bn(self.out_l1))

class NeuralNetDualNorm(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetDualNorm,self).__init__()
        self.l1_weight=Parameter(torch.Tensor(hidden_size,input_size))
        self.ll_bias=Parameter(torch.Tensor(hidden_size))
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(num_features=hidden_size)

        self.fc1 =nn.Linear(input_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,num_classes)

        self.b_ = Parameter(torch.Tensor(hidden_size))
        self.gamma_ = Parameter(torch.Tensor(hidden_size))
        self.c_ = Parameter(torch.Tensor(hidden_size))        

    def forward(self,x):
        self.out_l1=self.relu(self.fc1(x)) - self.b_
        # need to debug size because of batch
        self.out_last= self.fc2(self.out_rl1)
        return self.out_last

class DualParaNet(nn.Module):
    def __init__(self,neural_size):
        super(DualParaNet,self).__init__()
        self.neural_size=neural_size
        self.lambda_ = Parameter(torch.Tensor(neural_size))
        self.lambda_.data.fill_(0.5)
        self.mu_=Parameter(torch.Tensor(neural_size))
        self.mu_.data.fill_(1)
    
    def forward(self,x):
        variance_x=x*x
        return(
            self.lambda_*(variance_x-1)+\
            self.mu_*x)