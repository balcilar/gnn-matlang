
from torch_geometric.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (GINConv,global_mean_pool,GATConv,ChebConv,GCNConv)
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import MutagDataset,SpectralDesign

torch.manual_seed(0)
  
transform = SpectralDesign(nmax=28,adddegree=True,recfield=1,dv=4,nfreq=3) 
dataset = MutagDataset(root="dataset/mutag/",pre_transform=transform)


class PPGN(nn.Module):
    def __init__(self,nmax=28,nneuron=32):
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

        self.h1 = torch.nn.Linear(1*3*nneuron, 32)
        self.h2 = torch.nn.Linear(32, 1)


    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True)           

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 
        # read out mean or add ? just diagonal or diag and offdiag        
        xo1=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 
        # read out         
        xo2=torch.sum(x*data.M[:,0:1,:,:],(2,3))


        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # read out        
        xo3=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        x=torch.cat([xo1,xo2,xo3],1)         
        x=F.relu(self.h1(x))
        return self.h2(x)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(dataset.num_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64)

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)

        
        self.fc1 = torch.nn.Linear(64, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):

        x=data.x            
        edge_index=data.edge_index

        
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)        
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x) 
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)          

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 32, cached=False)
        self.conv2 = GCNConv(32, 64, cached=False)
        self.conv3 = GCNConv(64, 64, cached=False)       
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
             
        x = F.relu(self.conv2(x, edge_index))
        
        x = F.relu(self.conv3(x, edge_index)) 

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.conv1 = torch.nn.Linear(dataset.num_features, 32)
        self.conv2 = torch.nn.Linear(32, 32)
        self.conv3 = torch.nn.Linear(32, 32)      
        
        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))             
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x)) 
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=3
        nn=32
        self.conv1 = ChebConv(dataset.num_features, nn,S)
        self.conv2 = ChebConv(nn, nn, S)
        self.conv3 = ChebConv(nn, nn, S)
        
        self.fc1 = torch.nn.Linear(nn, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))              
        x = F.relu(self.conv2(x, edge_index))        
        x = F.relu(self.conv3(x, edge_index)) 

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 8, heads=8,concat=True, dropout=0.0)
        
        self.conv2 = GATConv(64, 16, heads=8, concat=True, dropout=0.0)

        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)

        self.fc1 = torch.nn.Linear(128, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):
        x=data.x         
        
        x = F.elu(self.conv1(x, data.edge_index))
        
        x = F.elu(self.conv2(x, data.edge_index))
        
        x = F.elu(self.conv3(x, data.edge_index)) 

        x = global_mean_pool(x, data.batch)        
        x = F.relu(self.fc1(x))        
        return self.fc2(x) 

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()

        S=1
        nout1=16
        nout2=32
        nout3=16
        nin=nout1+nout2+nout3

        self.bn1 = torch.nn.BatchNorm1d(nin)
        self.bn2 = torch.nn.BatchNorm1d(nin)
        self.bn3 = torch.nn.BatchNorm1d(nin)
        
        self.conv11 = SpectConv(dataset.num_features, nout2,S,selfconn=False)
        self.conv21 = SpectConv(nin, nout2, S,selfconn=False)
        self.conv31 = SpectConv(nin, nout2, S,selfconn=False)        
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout1)
        self.fc21 = torch.nn.Linear(nin, nout1)
        self.fc31 = torch.nn.Linear(nin, nout1)
        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc22 = torch.nn.Linear(nin, nout3)
        self.fc32 = torch.nn.Linear(nin, nout3)
        
        self.fc13 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc23 = torch.nn.Linear(nin, nout3)
        self.fc33 = torch.nn.Linear(nin, nout3)
       
        
        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1) 
       

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')        

        x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x))*F.relu(self.fc13(x))],1)
        x=self.bn1(x)
        
        x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x))*F.relu(self.fc23(x))],1)
        x=self.bn2(x)
        
        x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x))*F.relu(self.fc33(x))],1)
        x=self.bn3(x)

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=24
        nout2=24

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=False,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2) 
        self.bn1 = torch.nn.BatchNorm1d(nin)
        self.bn2 = torch.nn.BatchNorm1d(nin)
        self.bn3 = torch.nn.BatchNorm1d(nin)        
        
        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x=(self.conv1(x, edge_index,edge_attr))
        x=self.bn1(x)

        x=(self.conv2(x, edge_index,edge_attr))
        x=self.bn2(x)

        x=(self.conv3(x, edge_index,edge_attr))  
        x=self.bn3(x)       

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NB=np.zeros((500,10))

testsize=0
for fold in range(0,10):
    tsid=np.loadtxt('dataset/mutag/raw/10fold_idx/test_idx-'+str(fold+1)+'.txt')
    trid=np.loadtxt('dataset/mutag/raw/10fold_idx/train_idx-'+str(fold+1)+'.txt')
    trid=trid.astype(np.int)
    tsid=tsid.astype(np.int)

    bsize=16
    train_loader = DataLoader(dataset[[i for i in trid]], batch_size=bsize, shuffle=True)    
    test_loader  = DataLoader(dataset[[i for i in tsid]], batch_size=18, shuffle=False)
    
    model = GNNML3().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN GNNML1 GNNML3

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)         
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #,weight_decay=0.0001)

    trsize=trid.shape[0]    
    tssize=tsid.shape[0]

    testsize+=tssize

    def train(epoch):
        model.train()    
        L=0
        correct=0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            y_grd= (data.y) #.type(torch.long) 
            pre=model(data)
            pred=F.sigmoid(pre)
            #lss=F.nll_loss(pred, y_grd,reduction='sum')
            lss=F.binary_cross_entropy(pred[:,0], y_grd,reduction='sum')
            
            lss.backward()
            optimizer.step()
            
            correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

            L+=lss.item()
        return correct/trsize,L/trsize

    def test():
        model.eval()
        correct = 0
        L=0
        for data in test_loader:
            data = data.to(device)
            pre=model(data)
            pred=F.sigmoid(pre)
            y_grd= (data.y)
            correct += torch.round(pred[:,0]).eq(y_grd).sum().item()
            
            lss=F.binary_cross_entropy(pred[:,0], y_grd,reduction='sum')
            L+=lss.cpu().detach().numpy()

        s1= correct 
        
        return s1,L/tssize
       
    bval=1000
    btest=0
    for epoch in range(1, 101):
        tracc,trloss=train(epoch)
        test_acc,test_loss = test()     
        NB[epoch,fold]=test_acc   
        #print('Epoch: {:02d}, trloss: {:.4f},  Val: {:.4f}, Test: {:.4f}'.format(epoch,trloss,val_acc, test_acc))
        print('{:02d} Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Testloss: {:.4f}, Test acc: {:.4f}'.format(fold,epoch,trloss,tracc,test_loss,test_acc))
   
iter=NB.sum(1).argmax()
print((NB[iter,:]*100/18).mean())
print((NB[iter,:]*100/18).std())

