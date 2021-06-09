from torch_geometric.data import DataLoader
import torch
import scipy.io as sio
from torch_geometric.data.data import Data
import numpy as np

import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,global_add_pool,
                                global_mean_pool,GATConv,ChebConv,GCNConv)
from torch_geometric.datasets import MNISTSuperpixels
from libs.spect_conv import SpectConv,EdgeEncoder,SpectConCatConv
from libs.utils import PPGNAddDegree,get_n_params

transform = PPGNAddDegree(nmax=75,adddegree=True)  
train_dataset = MNISTSuperpixels('dataset/MNIST/', True, pre_transform=transform)
test_dataset = MNISTSuperpixels('dataset/MNIST/', False, pre_transform=transform)
train_loader = DataLoader(train_dataset[0:55000], batch_size=64, shuffle=True)
val_loader   = DataLoader(train_dataset[55000:60000], batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset[0:10000], batch_size=64, shuffle=False)

# normalize feature 
mx=train_dataset.data.x.max(0)
train_dataset.data.x=train_dataset.data.x/mx.values
test_dataset.data.x=test_dataset.data.x/mx.values

trsize=55000
tsize=10000
vsize=5000



class PPGN(nn.Module):
    def __init__(self,nmax=75,nneuron=64):
        super(PPGN, self).__init__()

        self.nmax=nmax        
        self.nneuron=nneuron
        ninp=train_dataset.data.X2.shape[1]
        
        bias=False
        self.mlp1_1 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_2 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_3 = torch.nn.Conv2d(nneuron+ninp, nneuron,1,bias=bias) 

        self.mlp2_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp2_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias) 

        self.mlp3_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)
        
        self.h1 = torch.nn.Linear(3*nneuron, 64) 
        self.h2 = torch.nn.Linear(64, 10)       
        

    def forward(self,data):
        x=data.X2 
        

        x1=F.relu(self.mlp1_1(x)) 
        x2=F.relu(self.mlp1_2(x))  
        x1x2 = torch.matmul(x1, x2)
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1)))
        # sum layer readout
        xo1=torch.sum(x*data.M[:,0:1,:,:],(2,3))
        

        x1=F.relu(self.mlp2_1(x)) 
        x2=F.relu(self.mlp2_2(x))  
        x1x2 = torch.matmul(x1, x2)
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1)))
        # sum layer readout       
        xo2=torch.sum(x*data.M[:,0:1,:,:],(2,3))
        

        x1=F.relu(self.mlp3_1(x)) 
        x2=F.relu(self.mlp3_2(x))  
        x1x2 = torch.matmul(x1, x2)
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1)))
        # sum  layer readout
        xo3=torch.sum(x*data.M[:,0:1,:,:],(2,3))


        x=torch.cat([xo1,xo2,xo3],1)                  
        x=F.relu(self.h1(x))
        x=self.h2(x)
        return F.log_softmax(x, dim=1)

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        nn=64
        self.conv1 = GCNConv(ninp, nn, cached=False)
        self.conv2 = GCNConv(nn, nn, cached=False)
        self.conv3 = GCNConv(nn, nn, cached=False)        
        self.bn1 = torch.nn.BatchNorm1d(nn)
        self.fc1 = torch.nn.Linear(nn, 32)
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x, edge_index))        
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) 
        x = global_mean_pool(x, data.batch)
        x=self.bn1(x)
        x = F.relu(self.fc1(x))        
        return F.log_softmax(self.fc2(x), dim=1)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        self.conv1 = GATConv(ninp, 8, heads=8, dropout=0.0)        
        self.conv2 = GATConv(8 * 8, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(8 * 16, 16, heads=8, concat=True, dropout=0.0)  
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc1 = torch.nn.Linear(128, 32)      
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):
        x=data.x       

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv3(x, data.edge_index)) 
        x = global_mean_pool(x, data.batch)

        x=self.bn1(x)

        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=5
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        self.conv1 = ChebConv(ninp, 64,S)
        self.conv2 = ChebConv(64, 64, S)
        self.conv3 = ChebConv(64, 64, S)

        self.bn1 = torch.nn.BatchNorm1d(64)
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, nout) #int(d.num_classes))

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        x = F.dropout(x, p=0.1, training=self.training)
        #x = F.relu(self.conv1(x, edge_index)) 
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = F.dropout(x, p=0.1, training=self.training)       
        #x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = F.dropout(x, p=0.1, training=self.training)
        #x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = global_mean_pool(x, data.batch)

        x=self.bn1(x)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1) 

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes

        nn1 = Sequential(Linear(ninp, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64) 

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)       
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)  
        x = F.relu(self.conv3(x, edge_index))
         
        x = global_mean_pool(x, data.batch)
        x = self.bn3(x)

        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        ninp=train_dataset.num_features
        nout=train_dataset.num_classes

        self.conv1 = torch.nn.Linear(ninp, 64)   
        self.conv2 = torch.nn.Linear(64, 64) 
        self.conv3 = torch.nn.Linear(64, 64) 
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.fc1 = torch.nn.Linear(64, 32)     
        self.fc2 = torch.nn.Linear(32, nout) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  

        x = global_mean_pool(x, data.batch)
        x=self.bn1(x)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()
        
        # number of neuron
        nout=64        
        # three part concatenate or sum?
        self.concat=False

        if self.concat:
            nin=3*nout
        else:
            nin=nout

        ninp=train_dataset.num_features
        self.conv11 = SpectConv(ninp,nout,selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)       
        self.bn1 = torch.nn.BatchNorm1d(nin)
        #self.bn2 = torch.nn.BatchNorm1d(nin)
        #self.bn3 = torch.nn.BatchNorm1d(nin)
        
        self.fc11 = torch.nn.Linear(ninp,nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)        
        
        self.fc12 = torch.nn.Linear(ninp,nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)        

        self.fc13 = torch.nn.Linear(ninp,nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)       
        
 
        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 10)
        

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')
        
        if self.concat:      
            x = F.dropout(x, p=0.1, training=self.training)      
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x)*self.fc13(x))],1)
            
            x = F.dropout(x, p=0.1, training=self.training)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x)*self.fc23(x))],1)
            
            x = F.dropout(x, p=0.1, training=self.training)
            x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x)*self.fc33(x))],1)
            
        else: 
            x = F.dropout(x, p=0.1, training=self.training)          
            x = F.relu(self.fc11(x)+self.conv11(x, edge_index,edge_attr)+self.fc12(x)*self.fc13(x)) # )+ torch.tanh(
            
            
            x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(self.fc21(x)+self.conv21(x, edge_index,edge_attr)+self.fc22(x)*self.fc23(x))
            #x=self.bn2(x)
            
            x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(self.fc31(x)+self.conv31(x, edge_index,edge_attr)+self.fc32(x)*self.fc33(x))
            #x=self.bn3(x)
                  

        x = global_mean_pool(x, data.batch)        
        x=self.bn1(x)

        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
        


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNML1().to(device)   #  GcnNet  GatNet  ChebNet GinNet MlpNet PPGN GNNML1

print(get_n_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    
    L=0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        lss=F.nll_loss(pred, data.y,reduction='sum')        
        lss.backward()
        L+=lss.cpu().detach().numpy()
        optimizer.step()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s1= correct / trsize
    return L/trsize,s1

def test():
    model.eval()
    correct = 0
    Lt=0
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        
        lss=F.nll_loss(pred, data.y,reduction='sum')
        Lt+=lss.cpu().detach().numpy()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s1= correct / tsize
    Lt=Lt/tsize
    correct = 0
    Lv=0
    for data in val_loader:
        data = data.to(device)
        pred = model(data)
        
        lss=F.nll_loss(pred, data.y,reduction='sum')
        Lv+=lss.cpu().detach().numpy()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s2= correct / vsize
    Lv=Lv/vsize
    return s1,Lt,s2,Lv

bval=0
btest=0
for epoch in range(1, 3001):
    trloss,tr_acc=train(epoch)
    test_acc,tloss,val_acc,vloss = test()
    if bval<val_acc:
        bval=val_acc
        btest=test_acc
    print('Epoch: {:02d}, train: {:.4f},{:.4f}, Val: {:.4f},{:.4f}, Test: {:.4f}, {:.4f} besttest:{:.4f} '.format(epoch,trloss,tr_acc,vloss,val_acc,tloss, test_acc,btest))
