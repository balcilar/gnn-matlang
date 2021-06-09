from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class SpectConv(MessagePassing):
    r"""
    """
    def __init__(self, in_channels, out_channels, K=1, selfconn=True, depthwise=False,bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpectConv, self).__init__(**kwargs)

        assert K > 0       

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise=depthwise
        #self.selfmult=selfmult
        
        self.selfconn=selfconn 
        
        
        if self.selfconn:
            K=K+1

        if self.depthwise:            
            self.DSweight = Parameter(torch.Tensor(K,in_channels))            
            self.nsup=K
            K=1
        

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        if self.depthwise:
            zeros(self.DSweight)

    def forward(self, x, edge_index,edge_attr, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        
        Tx_0 = x  
        out=0
        if not self.depthwise:
            enditr=self.weight.size(0)
            if self.selfconn:
                out = torch.matmul(Tx_0, self.weight[-1]) 
                enditr-=1 

            for i in range(0,enditr):
                h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None) 
                # if self.selfmult:
                #     h=h*Tx_0                
                out = out+ torch.matmul(h, self.weight[i])
        else:
            enditr=self.nsup
            if self.selfconn:
                out = Tx_0* self.DSweight[-1] 
                enditr-=1 

            out= out + (1+self.DSweight[0:1,:])*self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,0], size=None)
            for i in range(1,enditr):
                out= out + self.DSweight[i:i+1,:]*self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None) 

            out = torch.matmul(out, self.weight[0])                   

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))

class SpectConCatConv(MessagePassing):
    r"""
    """
    def __init__(self, in_channels, out_channels, K, selfconn=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpectConCatConv, self).__init__(**kwargs)

        assert K > 0       

        self.in_channels = in_channels
        self.out_channels = out_channels       
        
        self.selfconn=selfconn 
        
        
        if self.selfconn:
            K=K+1       
        

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(K*out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)        

    def forward(self, x, edge_index,edge_attr, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        
        Tx_0 = x  
        out=[]
        
        enditr=self.weight.size(0)
        if self.selfconn:
            out.append(torch.matmul(Tx_0, self.weight[-1])) 
            enditr-=1 

        for i in range(0,enditr):
            h = self.propagate(edge_index, x=Tx_0, norm=edge_attr[:,i], size=None)                           
            out.append(torch.matmul(h, self.weight[i]))

        out=torch.cat(out,1)               

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))


class EdgeEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()
        
        self.fc1 = torch.nn.Linear(emb_dim[0], emb_dim[1])
        self.fc2 = torch.nn.Linear(emb_dim[1], emb_dim[2])

    def forward(self, edge_attr):
        x = F.relu(self.fc1(edge_attr))
        x = F.relu(self.fc2(x))
        return x


class ML3Layer(torch.nn.Module):
    
    def __init__(self, learnedge,nedgeinput,nedgeoutput,ninp,nout1,nout2):
        super(ML3Layer, self).__init__()

        self.learnedge=learnedge
        self.nout2=nout2

        if self.learnedge:
            self.fc1_1 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_2 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_3 = torch.nn.Linear(nedgeinput, 2*nedgeinput,bias=False)
            self.fc1_4 = torch.nn.Linear(4*nedgeinput,nedgeoutput,bias=False)
        else:
            nedgeoutput=nedgeinput
        
        self.conv1 = SpectConv(ninp,nout1, nedgeoutput,selfconn=False)

        if nout2>0:
            self.fc11 = torch.nn.Linear(ninp, nout2) 
            self.fc12 = torch.nn.Linear(ninp, nout2)

    def forward(self, x,edge_index,edge_attr):
        if self.learnedge:
            tmp=torch.cat([F.relu(self.fc1_1(edge_attr)), torch.tanh(self.fc1_2(edge_attr))*torch.tanh(self.fc1_3(edge_attr))],1)
            edge_attr = F.relu(self.fc1_4(tmp))        
        if self.nout2>0:            
            x=torch.cat([F.relu(self.conv1(x, edge_index,edge_attr)), torch.tanh(self.fc11(x))*torch.tanh(self.fc12(x))],1)
        else:
            x=F.relu(self.conv1(x, edge_index,edge_attr))
        return x