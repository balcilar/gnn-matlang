
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GINConv,global_add_pool,GATConv,ChebConv,GCNConv)
import numpy as npi
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import GraphCountDataset,SpectralDesign,get_n_params
import scipy.io as sio
from sklearn.metrics import r2_score



transform = SpectralDesign(nmax=30,recfield=1,dv=1,nfreq=10,adddegree=True,laplacien=False,addadj=True)
dataset = GraphCountDataset(root="dataset/subgraphcount/",pre_transform=transform)

# normalize outputs
dataset.data.y=(dataset.data.y/dataset.data.y.std(0))
# normalize degree of the node
dataset.data.x[:,1]=dataset.data.x[:,1]/dataset.data.x[:,1].max()

a=sio.loadmat('dataset/subgraphcount/raw/randomgraph.mat')
trid=a['train_idx'][0]
vlid=a['val_idx'][0]
tsid=a['test_idx'][0]

train_loader = DataLoader(dataset[[i for i in trid]], batch_size=10, shuffle=True)
val_loader = DataLoader(dataset[[i for i in vlid]], batch_size=100, shuffle=False)
test_loader = DataLoader(dataset[[i for i in tsid]], batch_size=100, shuffle=False)



class PPGN(torch.nn.Module):
    def __init__(self,nmax=30,nneuron=40):
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

        self.mlp4_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp4_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp4_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)

        self.mlp5_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp5_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp5_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)
        
        self.h1 = torch.nn.Linear(2*5*nneuron, 64)
        self.h2 = torch.nn.Linear(64, 1)
        

    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True) 

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 

        # sum or mean layer readout
        xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        #xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)


        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 

        # sum or mean layer readout       
        xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        #xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # sum or mean layer readout
        xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        #xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)
        

        x1=F.relu(self.mlp4_1(x)*M) 
        x2=F.relu(self.mlp4_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp4_3(torch.cat([x1x2,x],1))*M) 

        # sum or mean layer readout
        xo4=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        #xo4=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1=F.relu(self.mlp5_1(x)*M) 
        x2=F.relu(self.mlp5_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp5_3(torch.cat([x1x2,x],1))*M) 

        # sum or mean layer readout
        xo5=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        #xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)


        x=torch.cat([xo1,xo2,xo3,xo4,xo5],1) 
        x=F.relu(self.h1(x))
        return self.h2(x)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(dataset.num_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)        

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)        

        nn4 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv4 = GINConv(nn4,train_eps=True)        

        nn5 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv5 = GINConv(nn5,train_eps=True)        
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x        
            
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        neuron=64
        self.conv1 = GCNConv(dataset.num_features, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False) 
        self.conv4 = GCNConv(neuron, neuron, cached=False) 
        self.conv5 = GCNConv(neuron, neuron, cached=False) 
        self.fc1 = torch.nn.Linear(neuron, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index  

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        neuron=64
        self.conv1 = torch.nn.Linear(dataset.num_features, neuron)
        self.conv2 = torch.nn.Linear(neuron, neuron)
        self.conv3 = torch.nn.Linear(neuron, neuron) 
        self.conv4 = torch.nn.Linear(neuron, neuron)
        self.conv5 = torch.nn.Linear(neuron, neuron)
        self.fc1 = torch.nn.Linear(neuron, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))                
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x)) 
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
class ChebNet(nn.Module):
    def __init__(self,S=5):
        super(ChebNet, self).__init__()
        neuron=48
        self.conv1 = ChebConv(dataset.num_features, neuron,S)
        self.conv2 = ChebConv(neuron, neuron, S)
        self.conv3 = ChebConv(neuron, neuron, S)   
        self.conv4 = ChebConv(neuron, neuron, S) 
        self.conv5 = ChebConv(neuron, neuron, S)      
        self.fc1 = torch.nn.Linear(neuron, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))              
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))        
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        x = F.relu(self.conv4(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        x = F.relu(self.conv5(x, edge_index,lambda_max=data.lmax,batch=data.batch))        

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 16, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv4 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv5 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.fc1 = torch.nn.Linear(128, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index        
        
        x = F.relu(self.conv1(x, edge_index))        
        x = F.relu(self.conv2(x, edge_index))        
        x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv4(x, edge_index)) 
        x = F.relu(self.conv5(x, edge_index)) 
        x = global_add_pool(x, data.batch)        
        x = F.relu(self.fc1(x))
        
        return self.fc2(x)

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()
        
        # number of neuron
        nout=32        
        # three part concatenate or sum?
        self.concat=True

        if self.concat:
            nin=3*nout
        else:
            nin=nout
        self.conv11 = SpectConv(dataset.num_features, nout,selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)
        self.conv41 = SpectConv(nin, nout, selfconn=False)
        self.conv51 = SpectConv(nin, nout, selfconn=False)
        
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)
        self.fc41 = torch.nn.Linear(nin, nout)
        self.fc51 = torch.nn.Linear(nin, nout)
        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)
        self.fc42 = torch.nn.Linear(nin, nout)
        self.fc52 = torch.nn.Linear(nin, nout)

        self.fc13 = torch.nn.Linear(dataset.num_features, nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)
        self.fc43 = torch.nn.Linear(nin, nout)
        self.fc53 = torch.nn.Linear(nin, nout)
        
 
        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')
        
        if self.concat:
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x)*self.fc13(x))],1)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x)*self.fc23(x))],1)
            x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x)*self.fc33(x))],1)
            x = torch.cat([F.relu(self.fc41(x)), F.relu(self.conv41(x, edge_index,edge_attr)),F.relu(self.fc42(x)*self.fc43(x))],1)
            x = torch.cat([F.relu(self.fc51(x)), F.relu(self.conv51(x, edge_index,edge_attr)),F.relu(self.fc52(x)*self.fc53(x))],1)
        else:            
            x = F.relu(self.fc11(x)+self.conv11(x, edge_index,edge_attr))+torch.tanh(self.fc12(x)*self.fc13(x))
            x = F.relu(self.fc21(x)+self.conv21(x, edge_index,edge_attr))+torch.tanh(self.fc22(x)*self.fc23(x))
            x = F.relu(self.fc31(x)+self.conv31(x, edge_index,edge_attr))+torch.tanh(self.fc32(x)*self.fc33(x))
            x = F.relu(self.fc41(x)+self.conv41(x, edge_index,edge_attr))+torch.tanh(self.fc42(x)*self.fc43(x))
            x = F.relu(self.fc51(x)+self.conv51(x, edge_index,edge_attr))+torch.tanh(self.fc52(x)*self.fc53(x))
        

        x = global_add_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x)

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # it learns edge or uses spectral designed one directly
        learnedge=True

        # number of neuron for for part1 and part2
        nout1=16
        nout2=16

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv4=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv5=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)

        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x=(self.conv1(x, edge_index,edge_attr))
        x=(self.conv2(x, edge_index,edge_attr))
        x=(self.conv3(x, edge_index,edge_attr))
        x=(self.conv4(x, edge_index,edge_attr))
        x=(self.conv5(x, edge_index,edge_attr))
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# select your model
model = GNNML3().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN  GNNML1  GNNML3

# select task, 0: triangle, 1: tailed_triangle 2: star  3: 4-cycle  4:custom
ntask=0



print('number of parameters:',get_n_params(model))

# be sure PPGN's bias are initialized by zero
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)         
                
model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train(epoch):
    model.train()
    
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)
        
        lss= torch.square(pre- data.y[:,ntask:ntask+1]).sum() 
        
        lss.backward()
        optimizer.step()  
        L+=lss.item()

    return L/len(trid)

def test():
    model.eval()
    yhat=[]
    ygrd=[]
    L=0
    for data in test_loader:
        data = data.to(device)

        pre=model(data)
        yhat.append(pre.cpu().detach())
        ygrd.append(data.y[:,ntask:ntask+1].cpu().detach())
        lss= torch.square(pre- data.y[:,ntask:ntask+1]).sum()         
        L+=lss.item()
    yhat=torch.cat(yhat)
    ygrd=torch.cat(ygrd)
    testr2=r2_score(ygrd.numpy(),yhat.numpy())

    Lv=0
    for data in val_loader:
        data = data.to(device)
        pre=model(data)
        lss= torch.square(pre- data.y[:,ntask:ntask+1]).sum() 
        Lv+=lss.item()    
    return L/len(tsid), Lv/len(vlid),testr2

bval=1000
btest=0
btestr2=0
for epoch in range(1, 1001):
    trloss=train(epoch)
    test_loss,val_loss,testr2 = test()
    if bval>val_loss:
        bval=val_loss
        btest=test_loss
        btestr2=testr2
    
    print('Epoch: {:02d}, trloss: {:.6f},  Valloss: {:.6f}, Testloss: {:.6f}, best test loss: {:.6f}, bestr2:{:.6f}'.format(epoch,trloss,val_loss,test_loss,btest,btestr2))

