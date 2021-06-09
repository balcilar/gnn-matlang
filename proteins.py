
from torch_geometric.data import DataLoader,InMemoryDataset
import torch
import scipy.io as sio
from torch_geometric.data.data import Data
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut,to_networkx
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,global_add_pool,global_max_pool,
                                global_mean_pool,GATConv,ChebConv,GCNConv)

import pickle
import os
import matplotlib.pyplot as plt
from libs.spect_conv import SpectConv,ML3Layer
from math import comb
from libs.utils import ProteinsDataset, SpectralDesign
torch.manual_seed(0)


transform = SpectralDesign(nmax=0,adddegree=True,recfiled=1,dv=4,nfreq=3) 
dataset = ProteinsDataset(root="dataset/proteins/",pre_transform=transform,contfeat=False)


class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn=200
        nn1 = Sequential(Linear(dataset.num_features, nn), ReLU(), Linear(nn, nn))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(nn)

        nn2 = Sequential(Linear(nn, nn), ReLU(), Linear(nn, nn))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(nn)
        
        self.fc2 = torch.nn.Linear(nn*2, 2) 

    def forward(self, data):

        x=data.x        
            
        edge_index=data.edge_index

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)              

        x = torch.cat([global_add_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        #x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        nn=200
        self.conv1 = GCNConv(dataset.num_features, nn, cached=False)
        self.conv2 = GCNConv(nn, nn, cached=False)

        self.bn1 = torch.nn.BatchNorm1d(nn)
        self.bn2 = torch.nn.BatchNorm1d(nn)
        
        self.fc1 = torch.nn.Linear(2*nn, 100)
        self.fc2 = torch.nn.Linear(100, 2) #int(d.num_classes))

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv1(x, edge_index))  
        x=self.bn1(x)

        x = F.dropout(x, p=0.1, training=self.training)      
        x = F.relu(self.conv2(x, edge_index))
        x=self.bn2(x)

        # x = F.dropout(x, p=0.1, training=self.training)      
        # x = F.relu(self.conv3(x, edge_index))

        # x = F.dropout(x, p=0.1, training=self.training)      
        # x = F.relu(self.conv4(x, edge_index))
        
        
        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        nn=200
        self.conv1 = torch.nn.Linear(dataset.num_features, nn)
        self.conv2 = torch.nn.Linear(nn, nn)        
        
        #self.fc1 = torch.nn.Linear(2*nn, 100)
        self.fc2 = torch.nn.Linear(2*nn, 2) #int(d.num_classes))

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        x = F.dropout(x, p=0.1, training=self.training) 
        x = F.relu(self.conv1(x)) 
              
        x = F.dropout(x, p=0.1, training=self.training) 
        x = F.relu(self.conv2(x))       
        

        x = torch.cat([global_add_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        #x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=3
        nn=200
        self.conv1 = ChebConv(dataset.num_features, nn,S)
        self.conv2 = ChebConv(nn, nn, S)
        #self.conv3 = ChebConv(nn, nn, S)
        #self.conv4 = ChebConv(nn, nn, S)

        self.bn1 = torch.nn.BatchNorm1d(nn)
        self.bn2 = torch.nn.BatchNorm1d(nn)
        
        
        #self.fc1 = torch.nn.Linear(2*nn, 6)
        self.fc2 = torch.nn.Linear(2*nn, 2) #int(d.num_classes))

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch)) 
        x=self.bn1(x)

        x = F.dropout(x, p=0.2, training=self.training)       
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        x=self.bn2(x)

        # x = F.dropout(x, p=0.1, training=self.training)       
        # x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        # x = F.dropout(x, p=0.1, training=self.training)       
        # x = F.relu(self.conv4(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        

        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        #x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 16, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        #self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        #self.conv4 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        self.fc1 = torch.nn.Linear(128*2, 100) 
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, data):
        x=data.x
        
                            
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        #x=self.bn1(x)

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        #x=self.bn2(x)  

        # x = F.dropout(x, p=0.1, training=self.training)
        # x = F.elu(self.conv3(x, data.edge_index))  

        # x = F.dropout(x, p=0.1, training=self.training)
        # x = F.elu(self.conv4(x, data.edge_index))        
        

        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()

        S=1
        nout1=64
        nout2=64
        nout3=16
        nin=nout1+nout2+nout3

        self.bn1 = torch.nn.BatchNorm1d(nin)
        self.bn2 = torch.nn.BatchNorm1d(nin)
        
        
        self.conv11 = SpectConv(dataset.num_features, nout2,S,selfconn=False)
        self.conv21 = SpectConv(nin, nout2, S,selfconn=False)
             
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout1)
        self.fc21 = torch.nn.Linear(nin, nout1)
        
        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc22 = torch.nn.Linear(nin, nout3)
        
        
        self.fc13 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc23 = torch.nn.Linear(nin, nout3)
        
       
        self.fc2 = torch.nn.Linear(2*nin,2) 
       

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')        
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x))*F.relu(self.fc13(x))],1)
        #x=self.bn1(x)
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x))*F.relu(self.fc23(x))],1)
        #x=self.bn2(x) 

        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        #x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=64
        nout2=0

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)        
        
        self.fc2 = torch.nn.Linear( 2*nin,2)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x = F.dropout(x, p=0.1, training=self.training)
        x=(self.conv1(x, edge_index,edge_attr))

        x = F.dropout(x, p=0.1, training=self.training)
        x=(self.conv2(x, edge_index,edge_attr))
        
        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)        
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
NB=np.zeros((500,10))

testsize=0
for fold in range(0,10):

    
    tsid=np.loadtxt('dataset/proteins/raw/10fold_idx/test_idx-'+str(fold+1)+'.txt')
    trid=np.loadtxt('dataset/proteins/raw/10fold_idx/train_idx-'+str(fold+1)+'.txt')
    trid=trid.astype(np.int)
    tsid=tsid.astype(np.int)

    ds=dataset.copy()
    d=dataset[[i for i in trid]].copy()
    ds.data.x=(ds.data.x-d.data.x.mean(0))/d.data.x.std(0)
    mn=d.data.x.mean(0)
    st=d.data.x.std(0)

    bsize=180
    train_loader = DataLoader(ds[[i for i in trid]], batch_size=bsize, shuffle=True)    
    test_loader  = DataLoader(ds[[i for i in tsid]], batch_size=60, shuffle=False)

    
    model = GNNML3().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  GNNML1 GNNML3       

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    trsize=trid.shape[0]    
    tssize=tsid.shape[0]

    testsize+=tssize

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
            optimizer.step()
            L+=lss.cpu().detach().numpy()
            pred = pred.max(1)[1]
            correct += pred.eq(data.y).sum().item()
        
        return correct/trsize,L/trsize


    def test():
        model.eval()
        correct = 0
        L=0
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            
            lss=F.nll_loss(pred, data.y,reduction='sum')
            L+=lss.cpu().detach().numpy()
            pred = pred.max(1)[1]
            correct += pred.eq(data.y).sum().item()
                
        return correct,L/tssize
       
    bval=1000
    btest=0
    for epoch in range(1, 51):
        tracc,trloss=train(epoch)
        test_acc,test_loss = test()     
        NB[epoch,fold]=test_acc   
        #print('Epoch: {:02d}, trloss: {:.4f},  Val: {:.4f}, Test: {:.4f}'.format(epoch,trloss,val_acc, test_acc))
        print('{:02d} Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Testloss: {:.4f}, Test acc: {:.4f}'.format(fold,epoch,trloss,tracc,test_loss,test_acc))

    print(NB.sum(1).max()/testsize)

import pandas as pd
pd.DataFrame(NB).to_csv('protein')
print(NB.sum(1).max()/testsize)
plt.plot(NB.sum(1));plt.show()
a=1