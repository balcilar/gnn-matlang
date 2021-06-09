
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GATConv,ChebConv,GCNConv,GINConv)

import numpy as np
import matplotlib.pyplot as plt
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import TwoDGrid30,SpectralDesign
from sklearn.metrics import r2_score



transform = SpectralDesign(nmax=900,recfield=5,dv=10,nfreq=10,adddegree=False)
dataset = TwoDGrid30(root="dataset/TwoDGrid30/",pre_transform=transform)

train_loader = DataLoader(dataset[0:1], batch_size=100000, shuffle=False)
test_loader = DataLoader(dataset[1:2], batch_size=100000, shuffle=False)
val_loader = DataLoader(dataset[2:3], batch_size=100000, shuffle=False)




class PPGN(nn.Module):
    def __init__(self,nmax=900,nneuron=20):
        super(PPGN, self).__init__()

        self.nmax=nmax        
        self.nneuron=nneuron
        ninp=dataset.data.X2.shape[1]
        
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
        
        self.h1 = torch.nn.Linear(2*3*nneuron, 1)        
        

    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True) 

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 

        # sum layer readout
        xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)
        

        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 

        # sum layer readout       
        xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)
        
        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # sum  layer readout
        xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2)),torch.sum(x*data.M[:,1:2,:,:],(2))],1)
        

        x=torch.cat([xo1,xo2,xo3],1) 
        x=torch.transpose(x[0,:,:],0,1)
        
        return self.h1(x)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(dataset.num_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)        

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)  
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x        
            
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index))        
        
        return self.fc2(x) 

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        neuron=64
        self.conv1 = GCNConv(dataset.num_features, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False) 
                
        self.fc2 = torch.nn.Linear(neuron, 1)
        

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index  

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) 
                
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        neuron=64
        self.conv1 = torch.nn.Linear(dataset.num_features, neuron)
        self.conv2 = torch.nn.Linear(neuron, neuron)
        self.conv3 = torch.nn.Linear(neuron, neuron)         
        
        self.fc2 = torch.nn.Linear(neuron, 1)
        
    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))                
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x)) 
                
        return self.fc2(x)
        
class ChebNet(nn.Module):
    def __init__(self,S=7):
        super(ChebNet, self).__init__()
        neuron=64
        self.conv1 = ChebConv(dataset.num_features, neuron,S)
        self.conv2 = ChebConv(neuron, neuron, S)
        self.conv3 = ChebConv(neuron, neuron, S) 

        self.fc2 = torch.nn.Linear(neuron, 1)
        
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))              
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))        
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch)) 

        
        return self.fc2(x)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 16, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
                
        self.fc2 = torch.nn.Linear(128, 1)
        

    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index        
        
        x = F.elu(self.conv1(x, edge_index))        
        x = F.elu(self.conv2(x, edge_index))        
        x = F.elu(self.conv3(x, edge_index)) 
                
        
        return self.fc2(x)

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()
        
        # number of neuron
        nout=32        
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
         
        self.fc2 = torch.nn.Linear(nin, 1)
        

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')
        
        if self.concat:
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x)*self.fc13(x))],1)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x)*self.fc23(x))],1)
            x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x)*self.fc33(x))],1)
            
        else:            
            x = F.relu(self.fc11(x))+ F.relu(self.conv11(x, edge_index,edge_attr))+F.relu(self.fc12(x)*self.fc13(x))
            x = F.relu(self.fc21(x))+ F.relu(self.conv21(x, edge_index,edge_attr))+F.relu(self.fc22(x)*self.fc23(x))
            x = F.relu(self.fc31(x))+ F.relu(self.conv31(x, edge_index,edge_attr))+F.relu(self.fc32(x)*self.fc33(x))
        
        return self.fc2(x)

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=32
        nout2=16

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)               
        
        self.fc2 = torch.nn.Linear(nin, 1)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x=(self.conv1(x, edge_index,edge_attr))
        x=(self.conv2(x, edge_index,edge_attr))
        x=(self.conv3(x, edge_index,edge_attr))        
                
        return self.fc2(x)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# select your model
model = GNNML3().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN  GNNML1  GNNML3


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  

model.apply(weights_init)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ntask bandpass:0, lowpass:1, highpass:2  
ntask=0

def visualize(tensor,n=30):
    y=tensor.detach().cpu().numpy()
    y=np.reshape(y,(n,n))
    plt.imshow(y.T);plt.colorbar();plt.show()


def train(epoch):
    model.train()
    ns=0
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)        
        lss= torch.square(data.mask*(pre- data.y[:,ntask:ntask+1])).sum() 
        lss.backward()
        optimizer.step()

        a=pre[data.mask==1]    
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1] 
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())
        L+=lss.item()
        # if you want to see the image that GNN  produce
        # visualize(pre)
    return L,r2

def test():
    model.eval()
    yhat=[]
    ygrd=[]
    L=0;vL=0
    for data in test_loader:
        data = data.to(device)
        optimizer.zero_grad()        
        pre=model(data)        
        lss= torch.square(data.mask*(pre- data.y[:,ntask:ntask+1])).sum() 
        L+=lss.item()
        a=pre[data.mask==1]    
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1] 
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())

    for data in val_loader:
        data = data.to(device)
        optimizer.zero_grad()        
        pre=model(data)        
        lss= torch.square(data.mask*(pre- data.y[:,ntask:ntask+1])).sum() 
        vL+=lss.item()
        a=pre[data.mask==1]    
        b=data.y[:,ntask:ntask+1] 
        b=b[data.mask==1] 
        vr2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())        

        # if you want to see the image that GNN  produce
        # visualize(pre)
    return L,r2,vL,vr2

bval=0
btest=0
for epoch in range(1, 2001):
    trloss ,tr2   =train(epoch)
    test_loss,r2,vallos,vr2= test()

    if bval<vr2:
        bval=vr2
        btest=r2
   
    print('Epoch: {:02d}, trloss: {:.4f}, r2: {:.4f},valloss: {:.4f}, valr2: {:.4f},testloss: {:.4f}, bestestr2: {:.8f},{:.8f}'.format(epoch,trloss,tr2,vallos,vr2,test_loss,r2,btest))
    