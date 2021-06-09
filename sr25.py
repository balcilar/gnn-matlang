
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GINConv,global_add_pool,GATConv,ChebConv,GCNConv)
import numpy as np

from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import SRDataset,SpectralDesign


transform = SpectralDesign(nmax=25,recfield=1,dv=2,nfreq=5,adddegree=True)
dataset = SRDataset(root="dataset/sr25/",pre_transform=transform)
train_loader = DataLoader(dataset, batch_size=100, shuffle=False)

class PPGN(torch.nn.Module):
    def __init__(self,nmax=25,nneuron=32):
        super(PPGN, self).__init__()

        self.nmax=nmax        
        self.nneuron=nneuron
        ninp=dataset.data.X2.shape[1]
        
        bias=True
        self.mlp1_1 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_2 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_3 = torch.nn.Conv2d(nneuron+ninp, nneuron,1,bias=bias) 

        self.mlp2_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp2_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias) 

        self.mlp3_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias) 
        
        self.h1 = torch.nn.Linear(2*3*nneuron, 10)
        

    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True) 

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 
        xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)


        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M)        
        xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)


        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 
        xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)

        
        x=torch.cat([xo1,xo2,xo3],1)  
        x=torch.sum(x,2)              
        x=torch.tanh(self.h1(x))
        return x

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()
        neuron=64
        r1=np.random.uniform()
        r2=np.random.uniform()
        r3=np.random.uniform()

        nn1 = Sequential(Linear(dataset.num_features, neuron))
        self.conv1 = GINConv(nn1,eps=r1,train_eps=True)        

        nn2 = Sequential(Linear(neuron, neuron))
        self.conv2 = GINConv(nn2,eps=r2,train_eps=True)        

        nn3 = Sequential(Linear(neuron, neuron))
        self.conv3 = GINConv(nn3,eps=r3,train_eps=True) 
        
        self.fc1 = torch.nn.Linear(neuron, 10)
        

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index
        
        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))        
        x = torch.tanh(self.conv3(x, edge_index))             

        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        neuron=64
        self.conv1 = GCNConv(dataset.num_features, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False) 
        self.fc1 = torch.nn.Linear(neuron, 10)
        

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index   

        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))
        x = torch.tanh(self.conv3(x, edge_index)) 
        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        neuron=64
        self.conv1 = torch.nn.Linear(dataset.num_features, neuron)
        self.conv2 = torch.nn.Linear(neuron, neuron)
        self.conv3 = torch.nn.Linear(neuron, neuron) 
        self.fc1 = torch.nn.Linear(neuron, 10)
        
    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = torch.tanh(self.conv1(x))                
        x = torch.tanh(self.conv2(x))        
        x = torch.tanh(self.conv3(x)) 
        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x
        
class ChebNet(nn.Module):
    def __init__(self,S=4):
        super(ChebNet, self).__init__()

        self.conv1 = ChebConv(dataset.num_features, 32,S)
        self.conv2 = ChebConv(32, 64, S)
        self.conv3 = ChebConv(64, 64, S)        
        self.fc1 = torch.nn.Linear(64, 10)
        
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        
        x = torch.tanh(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))              
        x = torch.tanh(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))        
        x = torch.tanh(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 16, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.fc1 = torch.nn.Linear(128, 10)
        

    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index        
        
        x = torch.tanh(self.conv1(x, edge_index))        
        x = torch.tanh(self.conv2(x, edge_index))        
        x = torch.tanh(self.conv3(x, edge_index)) 
        x = global_add_pool(x, data.batch)        
        x = torch.tanh(self.fc1(x))
        
        return x

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
        self.conv11 = SpectConv(dataset.num_features, nout,selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)
        
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)
        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)

        self.fc13 = torch.nn.Linear(dataset.num_features, nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)
        
 
        self.fc1 = torch.nn.Linear(nin, 10)
        

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cpu')
        
        if self.concat:
            x = torch.cat([torch.tanh(self.fc11(x)), torch.tanh(self.conv11(x, edge_index,edge_attr)),torch.tanh(self.fc12(x)*self.fc13(x))],1)
            x = torch.cat([torch.tanh(self.fc21(x)), torch.tanh(self.conv21(x, edge_index,edge_attr)),torch.tanh(self.fc22(x)*self.fc23(x))],1)
            x = torch.cat([torch.tanh(self.fc31(x)), torch.tanh(self.conv31(x, edge_index,edge_attr)),torch.tanh(self.fc32(x)*self.fc33(x))],1)
        else:            
            x = torch.tanh(self.fc11(x)+self.conv11(x, edge_index,edge_attr)+self.fc12(x)*self.fc13(x))
            x = torch.tanh(self.fc21(x)+self.conv21(x, edge_index,edge_attr)+self.fc22(x)*self.fc23(x))
            x = torch.tanh(self.fc31(x)+self.conv31(x, edge_index,edge_attr)+self.fc32(x)*self.fc33(x))
        

        x = global_add_pool(x, data.batch)
        x = self.fc1(x)
        return x


class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=32
        nout2=16

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2) 

        self.fc1 = torch.nn.Linear(nin, 10)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x=(self.conv1(x, edge_index,edge_attr))
        x=(self.conv2(x, edge_index,edge_attr))
        x=(self.conv3(x, edge_index,edge_attr))  

        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x


M=0
for iter in range(0,10):
    torch.manual_seed(iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # select your model
    model = PPGN().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN  GNNML1  GNNML3

    embeddings=[]
    model.eval()
    for data in train_loader:
        data = data.to(device)
        pre=model(data)
        embeddings.append(pre)

    E=torch.cat(embeddings).cpu().detach().numpy()    
    M=M+1*((np.abs(np.expand_dims(E,1)-np.expand_dims(E,0))).sum(2)>0.001)
    sm=((M==0).sum()-M.shape[0])/2
    print('similar:',sm)

    