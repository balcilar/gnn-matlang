import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import numpy as np
import networkx as nx
import pickle
import os
import scipy.io as sio
from math import comb


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class PtcDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        
        super(PtcDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["ptc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'].astype(np.int)
        Y=Y[:,0]

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i]).type(torch.float32)                 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(ProteinsDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["proteins.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'].astype(np.int)
        Y=Y[:,0]

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            if self.contfeat:
                #ind=list(set(range(0,F[i].shape[1]))-set([3,4]))
                tmp=F[i]#[:,ind]
                x=torch.tensor(tmp).type(torch.float32)
            else:
                x=torch.tensor(F[i][:,0:3]).type(torch.float32) 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class EnzymesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(EnzymesDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["enzymes.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'][0].astype(np.int)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            if self.contfeat:
                #ind=list(set(range(0,F[i].shape[1]))-set([3,4]))
                tmp=F[i]#[:,ind]
                x=torch.tensor(tmp).type(torch.float32)
            else:
                x=torch.tensor(F[i][:,0:3]).type(torch.float32) 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class MutagDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MutagDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["mutag.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        TA=a['TA'][0]
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=((a['y']+1)//2).astype(np.float32)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i]).type(torch.float32) 
            y=torch.tensor(Y[i]).type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Zinc12KDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Zinc12KDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Zinc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        F=a['F'][0]
        A=a['E'][0]
        Y=a['Y']
        nmax=37
        ntype=21
        maxdeg=4

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.zeros(A[i].shape[0],ntype+maxdeg)
            deg=(A[i]>0).sum(1)
            for j in range(F[i][0].shape[0]):
                # put atom code
                x[j,F[i][0][j]]=1
                # put degree code
                x[j,-int(deg[j])]=1
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class BandClassDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BandClassDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["bandclass.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        F=a['F']
        Y=a['Y']
        F=np.expand_dims(F,2)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i,:,:]) 
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class TwoDGrid30(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid30, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["TwoDGrid30.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        F=F.astype(np.float32)

        data_list = []
        E=np.where(A>0)
        edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        x=torch.tensor(F[:,0:1])
        y=torch.tensor(F[:,1:4])
        mask=torch.tensor(F[:,12:13])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask))
        x=torch.tensor(F[:,4:5])
        y=torch.tensor(F[:,5:8])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask)) 
        x=torch.tensor(F[:,8:9])
        y=torch.tensor(F[:,9:12])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SpectralDesign(object):   

    def __init__(self,nmax=0,recfield=1,dv=5,nfreq=5,adddegree=False,laplacien=True,addadj=False,vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield=recfield  
        # b parameter
        self.dv=dv
        # number of sampled point of spectrum
        self.nfreq=  nfreq
        # if degree is added to node feature
        self.adddegree=adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien=laplacien
        # add adjacecny as edge feature
        self.addadj=addadj
        # use given max eigenvalue
        self.vmax=vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax=nmax    

    def __call__(self, data):

        n =data.x.shape[0]     
        nf=data.x.shape[1]  


        data.x=data.x.type(torch.float32)  
               
        nsup=self.nfreq+1
        if self.addadj:
            nsup+=1
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0],data.edge_index[1]]=1
        
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(A.sum(0)).unsqueeze(-1)],1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield==0:
            M=A
        else:
            M=(A+np.eye(n))
            for i in range(1,self.recfield):
                M=M.dot(M) 

        M=(M>0)

        
        d = A.sum(axis=0) 
        # normalized Laplacian matrix.
        dis=1/np.sqrt(d)
        dis[np.isinf(dis)]=0
        dis[np.isnan(dis)]=0
        D=np.diag(dis)
        nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
        V,U = np.linalg.eigh(nL) 
        V[V<0]=0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax=V.max().astype(np.float32)

        if not self.laplacien:        
            V,U = np.linalg.eigh(A)

        # design convolution supports
        vmax=self.vmax
        if vmax is None:
            vmax=V.max()

        freqcenter=np.linspace(V.min(),vmax,self.nfreq)
        
        # design convolution supports (aka edge features)         
        for i in range(0,len(freqcenter)): 
            SP[i,:,:]=M* (U.dot(np.diag(np.exp(-(self.dv*(V-freqcenter[i])**2))).dot(U.T))) 
        # add identity
        SP[len(freqcenter),:,:]=np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter)+1,:,:]=A
           
        # set convolution support weigths as an edge feature
        E=np.where(M>0)
        data.edge_index2=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)  

        # set tensor for Maron's PPGN         
        if self.nmax>0:       
            H=torch.zeros(1,nf+2,self.nmax,self.nmax)
            H[0,0,data.edge_index[0],data.edge_index[1]]=1 
            H[0,1,0:n,0:n]=torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0,nf):      
                H[0,j+2,0:n,0:n]=torch.diag(data.x[:,j])
            data.X2= H 
            M=torch.zeros(1,2,self.nmax,self.nmax)
            for i in range(0,n):
                M[0,0,i,i]=1
            M[0,1,0:n,0:n]=1-M[0,0,0:n,0:n]
            data.M= M        

        return data

class PPGNAddDegree(object):   

    def __init__(self,nmax=0,adddegree=True,):
        
        # if degree is added to node feature
        self.adddegree=adddegree       

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax=nmax    

    def __call__(self, data):

        n =data.x.shape[0]     
        nf=data.x.shape[1]  


        data.x=data.x.type(torch.float32)
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1
        
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(A.sum(0)).unsqueeze(-1)],1)
            
        # set tensor for Maron's PPGN         
        if self.nmax>0:       
            H=torch.zeros(1,nf+2,self.nmax,self.nmax)
            H[0,0,data.edge_index[0],data.edge_index[1]]=1 
            H[0,1,0:n,0:n]=torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0,nf):      
                H[0,j+2,0:n,0:n]=torch.diag(data.x[:,j])
            data.X2= H 
            M=torch.zeros(1,2,self.nmax,self.nmax)
            for i in range(0,n):
                M[0,0,i,i]=1
            M[0,1,0:n,0:n]=1-M[0,0,0:n,0:n]
            data.M= M        

        return data
    
class DegreeMaxEigTransform(object):   

    def __init__(self,adddegree=True,maxdeg=40,addposition=False,addmaxeig=True):
        self.adddegree=adddegree
        self.maxdeg=maxdeg
        self.addposition=addposition
        self.addmaxeig=addmaxeig

    def __call__(self, data):

        n=data.x.shape[0] 
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1         
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(1/self.maxdeg*A.sum(0)).unsqueeze(-1)],1)
        if self.addposition:
            data.x=torch.cat([data.x,data.pos],1)

        if self.addmaxeig:
            d = A.sum(axis=0) 
            # normalized Laplacian matrix.
            dis=1/np.sqrt(d)
            dis[np.isinf(dis)]=0
            dis[np.isnan(dis)]=0
            D=np.diag(dis)
            nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
            V,U = np.linalg.eigh(nL)               
            vmax=np.abs(V).max()
            # keep maximum eigenvalue for Chebnet if it is needed
            data.lmax=vmax.astype(np.float32)        
        return data    
    
