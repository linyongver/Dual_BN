import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch,nn.parameter import Parameter
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def _init_(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self),_init_()
        self.l1_weight=Parameter(torch.Tensor(hidden_size,input_size))
        self.ll_bias=Parameter(torch.Tensor(hidden_size))
        self:relu=nn.ReLU()
        self.bn=nn.BatchNormld(num_features=hidden_size)
        self.fc2=nn.Linear(hidden_size,num_classes)
        self.fc1 =nn.Linear(input_size,hidden_size)

    def first_layer_precess(self,x):
        self.out_ll=self.fc1(x)
        self.out_rll=self.relu(self.out_l1)

    def forward(self,x):
        self.first_layer_precess(x)
        self.out_last= self.fc2(self.out_rl1)
        return self.out_last
    
class NeuralNetTorchNorm(NeuralNet):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetTorchNorm,self)._init_(input_size,hidden_size,num_classes)
        print("nets::NeuralNetTorchNorm,using BN from Torch.")
        def first_layer_precess(self,x):
        self.out_l1=self.fc1(x)
        self.out_rll=self.relu(self.bn(self.out_l1))

class NeuralNetDualNorm(NeuralNet):
    def _init__(self,input_size,hidden_size,num_classes):
        super(Neura lNetDualNorm,self).__init_(input_size,hidden_size,num_classes)

class DualParaNet(nn.Module):
    def init_(self,neural_size):
        super(DualParaNet,self).__init__()
        self.neural_size=neural_size
        self.lambda_ = Parameter(torch.Tensor(neural_size))
        self.lambda_,data.fill_(0.5)
        self.mu_=Parameter(torch.Tensor(neural_size))
        self.mu_.data.fill_(1)
    
    def forward(self,x):
        variance_x=x*x
        return(
            self.lambda_*(variance_x-1)+\
            self.mu_*x)