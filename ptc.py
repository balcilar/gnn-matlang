
from torch_geometric.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (GINConv,global_add_pool,global_max_pool,global_mean_pool,GATConv,ChebConv,GCNConv)
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import PtcDataset, SpectralDesign

torch.manual_seed(1)



transform = SpectralDesign(nmax=109,adddegree=True,recfield=1,dv=10,nfreq=10) 
dataset = PtcDataset(root="dataset/PTC/",pre_transform=transform)


class PPGN(nn.Module):
    def __init__(self,nmax=109,nneuron=32):
        super(Maron2FWL, self).__init__()

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

        # self.mlp3_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        # self.mlp3_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        # self.mlp3_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)

        # self.mlp4_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        # self.mlp4_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        # self.mlp4_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias) 

        self.h1 = torch.nn.Linear(1*2*nneuron, 32)
        self.h2 = torch.nn.Linear(32, 6)


    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True)        

        #x = F.dropout(x, p=0.1, training=self.training)      

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 
        
        xo1=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        #x = F.dropout(x, p=0.1, training=self.training)

        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 
        
        xo2=torch.sum(x*data.M[:,0:1,:,:],(2,3))


        # x1=F.relu(self.mlp3_1(x)*M) 
        # x2=F.relu(self.mlp3_2(x)*M)  
        # x1x2 = torch.matmul(x1, x2)*M
        # x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # xo3=torch.sum(x*data.M[:,0:1,:,:],(2,3))


        # x1=F.relu(self.mlp4_1(x)*M) 
        # x2=F.relu(self.mlp4_2(x)*M)  
        # x1x2 = torch.matmul(x1, x2)*M
        # x=F.relu(self.mlp4_3(torch.cat([x1x2,x],1))*M)         
       
        # xo4=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        
        x=torch.cat([xo1,xo2],1) 

        x= F.relu(self.h1(x))        
        
        return F.log_softmax(self.h2(x), dim=1)

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
        self.conv3 = GCNConv(nn, nn, cached=False)
        self.conv4 = GCNConv(nn, nn, cached=False)

        self.bn1 = torch.nn.BatchNorm1d(nn)
        self.bn2 = torch.nn.BatchNorm1d(nn)
        
        self.fc1 = torch.nn.Linear(2*nn, 100)
        self.fc2 = torch.nn.Linear(100, 2) #int(d.num_classes))

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv1(x, edge_index))  
        #x=self.bn1(x)

        x = F.dropout(x, p=0.1, training=self.training)      
        x = F.relu(self.conv2(x, edge_index))
        #x=self.bn2(x)

        x = F.dropout(x, p=0.1, training=self.training)      
        x = F.relu(self.conv3(x, edge_index))

        x = F.dropout(x, p=0.1, training=self.training)      
        x = F.relu(self.conv4(x, edge_index))
        
        
        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        nn=64
        self.conv1 = torch.nn.Linear(dataset.num_features, nn)
        self.conv2 = torch.nn.Linear(nn, nn)        
        
        self.fc1 = torch.nn.Linear(2*nn, 100)
        self.fc2 = torch.nn.Linear(100, 2) #int(d.num_classes))

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        #x = F.dropout(x, p=0.1, training=self.training) 
        x = F.relu(self.conv1(x)) 
              
        #x = F.dropout(x, p=0.1, training=self.training) 
        x = F.relu(self.conv2(x))       
        

        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=3
        nn=200
        self.conv1 = ChebConv(dataset.num_features, nn,S)
        self.conv2 = ChebConv(nn, nn, S)
        self.conv3 = ChebConv(nn, nn, S)
        self.conv4 = ChebConv(nn, nn, S)

        self.bn1 = torch.nn.BatchNorm1d(nn)
        self.bn2 = torch.nn.BatchNorm1d(nn)
        
        
        #self.fc1 = torch.nn.Linear(2*nn, 6)
        self.fc2 = torch.nn.Linear(2*nn, 2) #int(d.num_classes))

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch)) 
        #x=self.bn1(x)

        x = F.dropout(x, p=0.2, training=self.training)       
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        #x=self.bn2(x)

        x = F.dropout(x, p=0.2, training=self.training)       
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = F.dropout(x, p=0.2, training=self.training)       
        x = F.relu(self.conv4(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        

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
        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        self.conv4 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)

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

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv3(x, data.edge_index))  

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv4(x, data.edge_index))        
        

        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()

        S=1
        nout1=32
        nout2=64
        nout3=2
        nin=nout1+nout2+nout3

        self.bn1 = torch.nn.BatchNorm1d(nin)
        self.bn2 = torch.nn.BatchNorm1d(nin)
        self.bn3 = torch.nn.BatchNorm1d(nin)
        self.bn4 = torch.nn.BatchNorm1d(nin)
        
        
        self.conv11 = SpectConv(dataset.num_features, nout2,S,selfconn=False)
        self.conv21 = SpectConv(nin, nout2, S,selfconn=False)       
             
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout1)
        self.fc21 = torch.nn.Linear(nin, nout1)        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc22 = torch.nn.Linear(nin, nout3)        
        self.fc13 = torch.nn.Linear(dataset.num_features, nout3)
        self.fc23 = torch.nn.Linear(nin, nout3)
        
        self.fc1 = torch.nn.Linear(2*nin,100) 
        self.fc2 = torch.nn.Linear(100,2) 
       

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.tanh(self.fc12(x))*F.tanh(self.fc13(x))],1)
        x=self.bn1(x)
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.tanh(self.fc22(x))*F.tanh(self.fc23(x))],1)
        #x=self.bn2(x) 
        

        x = torch.cat([global_add_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=64
        nout2=16

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)  
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin, nout1=nout1,nout2=nout2)
        self.conv4=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)       
        
        self.fc1 = torch.nn.Linear(2*nin, 100)
        self.fc2 = torch.nn.Linear( 100,2)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x = F.dropout(x, p=0.2, training=self.training)
        x=(self.conv1(x, edge_index,edge_attr))

        x = F.dropout(x, p=0.2, training=self.training)
        x=(self.conv2(x, edge_index,edge_attr))

        x = F.dropout(x, p=0.2, training=self.training)
        x=(self.conv3(x, edge_index,edge_attr))

        x = F.dropout(x, p=0.2, training=self.training)
        x=(self.conv4(x, edge_index,edge_attr))
        
        x = torch.cat([global_mean_pool(x, data.batch),global_max_pool(x, data.batch)],1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NB=np.zeros((500,10))

testsize=0
for fold in range(0,10):

    
    tsid=np.loadtxt('dataset/PTC/raw/10fold_idx/test_idx-'+str(fold+1)+'.txt')
    trid=np.loadtxt('dataset/PTC/raw/10fold_idx/train_idx-'+str(fold+1)+'.txt')
    trid=trid.astype(np.int)
    tsid=tsid.astype(np.int)

    ds=dataset.copy()
    # d=dataset[[i for i in trid]].copy()
    # ds.data.x=(ds.data.x-d.data.x.mean(0))/d.data.x.std(0)
    # mn=d.data.x.mean(0)
    # st=d.data.x.std(0)

              

    bsize=32
    train_loader = DataLoader(ds[[i for i in trid]], batch_size=bsize, shuffle=True)    
    test_loader  = DataLoader(ds[[i for i in tsid]], batch_size=60, shuffle=False)

    
    model = GNNML3().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN GNNML1 GNNML3 
    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)         
    model.apply(weights_init)


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
    for epoch in range(1, 101):
        tracc,trloss=train(epoch)
        test_acc,test_loss = test()     
        NB[epoch,fold]=test_acc   
        #print('Epoch: {:02d}, trloss: {:.4f},  Val: {:.4f}, Test: {:.4f}'.format(epoch,trloss,val_acc, test_acc))
        print('{:02d} Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Testloss: {:.4f}, Test ntrue: {:.4f}'.format(fold,epoch,trloss,tracc,test_loss,test_acc))

    print(NB.sum(1).max()/testsize)

mi=NB.sum(1).argmax()
print((NB[mi,:]*100/34).mean())
print((NB[mi,:]*100/34).std())
