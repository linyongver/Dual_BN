import argparse
import torch
import model
parser=argparse.ArgumentParser()
parser.add_argument("--n_epochs",type=int,default=200,help="number of epochs of training")
parser.add_argument("--random_seed",type=int,default=1,help="random seed for torch")
parser.add_argument("--hidden_size",type=int,default=500,help="hidden units")
parser.add_argument("--batch_size",type=int,default=64,help="size of the batches")
parser.add_argument("--lr",type=float,default=0.001,help="adam:learning rate")
parser.add_argument("--bn_type",type=str,default="no",help="chose from [no,torch_bn,dual_bn],the dn type")
opt=parser.parse_args()
print(opt)

def get_model(bn_type):
    if bn_type=="no":
        return model.NNmodel
    elif bn_type=="torch_bn":
        return model.NNTorchBNModel
    elif bn_type=="dual_bn":
        return model.NNDualBNModel
    else:
        raise NotImplementedError

torch.manual_seed(opt.random_seed)
model= get_model(opt.bn_type)(
    input_size=784,
    hidden_size=opt.hidden_size,
    num_classes=10)
model.train_model(
    num_epochs=opt.n_epochs,
    batch_size=opt.batch_size,
    learning_rate=opt.lr)
model.eval_acc(model.train_loader)