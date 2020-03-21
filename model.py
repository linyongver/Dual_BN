import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from utils import get_net

class BasicModel(object):
    def __init__(self):
        pass
    def load_data(self):
        pass
    def build_model(self):
        pass
    def train_model(self):
        pass
    def eval_acc(self,data_loader):
        pass
    def save_model(self):
        pass
    def register_result(self,names,values):
        pass
    def write_result(self):
        pass

class NNmodel(BasicModel):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NNmodel,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.device=torch.device("cpu")
        #'cuda'if torch.cuda,is_available()
        #else 'cpu'
        self.result_list=[]

    def load_data(self):
        self.train_dataset=torchvision.datasets.MNIST(
            root='../../data',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
        self.test_dataset=torchvision.datasets.MNIST(
            root='../../data',
            train=False,
            transform=transforms.ToTensor())
        self.train_loader=torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True)
        self.test_loader=torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False)

    def register_result(self, names, values):
        assert len(names)== len(values)
        result_dict =dict(zip(names, values))
        self.result_list.append(result_dict)
        self.write_result()

    def write_result(self):
        result_df =pd.DataFrame(self.result_list)
        result_df.to_csv("results/test.csv", index=False)

    def get_net_by_bn(self):
        net =get_net(bn_type="no")#net_class
        return net

    def build_model(self):
        self.net =self.get_net_by_bn()
        self.model=self.net(
            self.input_size,
            self.hidden_size,
            self.num_classes).to(self.device)
        self.criterion=nn.CrossEntropyLoss()
            self.optimizer=torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)

    def train_model(self, num_epochs,batch_size,learning_rate):
        # num_epochs=5
        # batch_size=100
        # learning_rate=0.001
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.load_data()
        self.build_model()
        total_step=len(self.train_loader)
        for epoch in range(num_epochs):
            for i,(images,labels)in enumerate(self.train_loader):
                # Move tensors to the configured device
                images= images.reshape(-1,28*28).to(self.device)
                labels= labels.to(self.device)
                #Forward pass
                outputs= self.model(images)
                self.loss=self.criterion(outputs,labels)
                # Backward and optimize
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                if(i+1)%100==0:
                print('Epoch [{}/{}],Step [{}/{}],Loss:{:.4f}'.format(epoch+1,num_epochs,i+1,total_step,self.loss.item()))
                self.register_result(
                    names=["epoch","num_epochs","step","total_step","loss","acc"],
                    values=[epoch+1,num_epochs,i+1,total step,round(self.loss.item(),4),np.nan])
  
    def eval_acc(self,data_loader):
        with torch.no_grad():
            correct=0
            total=0
            for images,labels in data_loader:
                self.images=images
                self.labels=labels
                images=images.reshape(-1,28*28).to(self.device)
                labels=labels.to(self.device)
                outputs= self.model(images)
                _, predicted= torch.max(outputs.data,1)
                total+= labels.size(0)
                correct+=(predicted==labels).sum().item()
                break
        return 100.0* correct/total

class NNTorchBNModel(NNmodel):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NNTorchBNModel,Iself).__init__(
        input_size,hidden_size,num_classes)

    def get_net_by_bn(self):
        net=get_net(bn_type="torch_bn")#net_class
        return net

class NNDualBNModel(NNmodel):
    def __init__(self,input_size,hidden_size,num_classes):
        Super(NNDualBNModel,self).__init__(
            input_size,hidden_size,num_classes)  

    def build_model(self):
        self.net,self.dual_net=get_net(bn_type="dual_bn")
        self.model=self.net(
            self.input_size,
            self.hidden_size,
            self.num_classes).to(self.device)
        self.dual_model=self.dual_net(
        self.hidden_size).to(self.device)
        self.criterion=nn.CrossEntropyLoss()
        self.optimizer_nn= torch.optim.SGD(
            self.model.parameters(),
            lr=self,learning_rate)
        self.optimizer_dual=torch.optim.SGD(
            self.dual_model.parameters(),
            lr=self.learning_rate* 10)

    def lr_shrink(self,shrink_rate):
        for i in range(len(self.optimizer_nn.param_groups)):
        self.optimizer_nn.param_groupslill'lr'] *= shrink_rate
        for i in range(len(self.optimizer_dual.param_groups)):
        self.optimizer_dual.param_groups[i]['lr'] *= shrink_rate
        # self.optimizer_nn = torch.optim.SGD(
        # self.model.parameters(),
        # lr=self.learning_rate* shrink_rate)
        # self.optimizer_dual=torch.optim.SGD(
        # self,dual_model.parameters(),
        # lr=self.learning_rate)

    def eval_distribution(self,data_loader):
        hidden_list=[]
        clip_num=0 #len(data_loader)//3
        def eval_distribution(self,data_loader):
            hidden_list=[]
            clip_num=0# len(data_loader)//3
            for i,(images,labels)in enumerate(data_loader):
                images=self.tmp_images
                labels= self.tmp_labels
                images=images.reshape(-1,28*28).to(self.device)
                labels=labels.to(self.device)
                outputs= self.model(images)
                epr_loss =self.criterion(outputs,labels)
                hidden_list.append(self.model.out_l1)
                if i>= clip_num:
                    break
                fullcat = torch.cat([x for x in hidden_list],dim=0)
                mean_var_df = pd.DataFrame({
                    "mean":torch.mean(fullcat,dim=0).data,
                    "var":torch.mean(fullcat**2,dim=0).data
                })
        return mean_var_df["mean"].mean(),mean_var_df["var"].mean()

    def train_model(self,num_epochs,batch_size,learning_rate):
        # num_epochs=5
        # batch _size=100
        # learning_rate=0.001
        self.num_epochs=num_epochs
        self.batch_size=batch size
        self.learning_rate=learning_rate
        self.load_data()
        self.build_model()
        total_step=len(self.train_loader)
        for i,(images,labels)in enumerate(self.train_loader):
            self.tmp_images=images
            self.tmp_labels=labels
            break
        for epoch in range(num_epochs):
            #if epoch== num_epochs*2//3:
            #   self.lr_shrink(shrink_rate=0.1)
            #   print("resetting learning rate by shrinking to 0.1")
            # Move tensors to the configured device
            for i,(_,_) in enumerate(self.train_loader):
                images = self.tmp_images
                labels= self.tmp_labels
                images =images.reshape(-1,28*28).to(self.device)
                labels=.labels.to(self.device)
                # Forward pass
                # outputs= self.model(images)
                # self.epr_loss= self.criterion(outputs,labels)
                # self.dual_loss= torch.mean(self.dual_model(

                outputs=self.model(images)
                self.epr_loss=self.criterion(outputs,labels)
                self.dual_loss=torch.mean(self.dual_model(
                    self.model.out_l1))
                self.loss=self.epr_loss#+self.dual_loss#self.epr_loss+
                self.neg_loss=-self.epr_loss#-self.dual_loss#-self.epr_loss
                # Backward(Neural Network)and optimize
                self.optimizer_nn.zero_grad()
                self.loss.backward(retain_graph=True)
                self.optimizer_nn.step()
                # Backward(Dual Parameter)and optimize
                # self.optimizer_dual.zero_grad()
                #self.neg_loss.backward()
                # self.optimizer_dual.step()
                if(i)%1==0:
                    self.register_result(
                        names=["type","dataset","epoch","num_epochs","step","total_step","loss"],
                        values=["train_loss","train",epoch+1,num_epochs,i+1,total_step,round(self.loss.item(),4)])
                    hidden_mean_list=(self.model.out_l1.mean(dim=0).data).numpy().tolist()
                    hidden_mean_names=["hidden_mean_%s"%i for i in range(len(hidden_mean_list))]
                    mu_list=(self.dual_model.mu_.data).numpy().tolist()
                    mu_names=["mu_%s"%i for i in range(len(mu_list))]
                    hidden_var_list=((self.model.out_l1**2).mean(dim=0).data).numpy().tolist()
                    hidden_var_names=["hidden_var_%s"%i for i in range(len(hidden_var_list))]
                    lambda_list=(self.dual_model.lambda_.data).numpy().tolist()
                    lamnbda_names=["lambda_%s"%i for i in range(len(lambda_list))]
                    hoop_names=["type","dataset","epoch","num_epochs","step","total_step"]
                    hoop_list=["hidden_state","train",epoch+1,num_epochs,i+1,total_step]
                    self.register_result(
                        names=hoop_names+hidden_mean_names+hidden_var_names+mu_names+lambda_names,
                        values=hoop_list+hidden_mean_list+hidden_var_list+mu_list+lambda_list)
                    mean,var=self.eval_distribution(
                        self.train_loader)
                    acc=self.eval_acc(self.test_loader)
                    self.register_result(
                        names=["'type","dataset","epoch","num_epochs","step","total_step","acc"],
                        values=["eval_acc","test",epoch+1,num_epochs,i+1,total_step,acc])
                    self.register_result(
                        names=["type","dataset","epoch","num_epochs","step","total_step","mean","var"],
                        values=["eval_dist","train",epoch+1,num_epochs,i+1,total_step,mean,var])
                    if i%10=0:
                        print(
                            'Epoch [{}/{}],Step [{/{}],Loss_epr:{:.4f},Loss_cnstr:{:.4f},'
                            .format(epoch+1,num_epochs,i+1,total_step,self.epr_loss,self.dual_loss.data))
                        print("mean=%.4f,var=%.4f,acc=%.4f"%(mean,var,acc))
                    self.write_result()