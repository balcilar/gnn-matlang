import numpy as np
from torch_geometric.datasets import MNISTSuperpixels
from libs.utils_tf import *
from libs.utils import DegreeMaxEigTransform
   
#select if node degree and location of superpixel region would be used by model or not.
#after any chnageing please remove MNIST/processed folder in order to preprocess changes again.
transform=DegreeMaxEigTransform(adddegree=True,addposition=False,addmaxeig=False)

train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True, pre_transform=transform)
test_dataset = MNISTSuperpixels(root='dataset/MNIST', train=False, pre_transform=transform)

# nkernel+1 supports of GNNML3 to be prepared
nkernel=5
# receptive field and bandwidth parameter
recfield=3
dv=10

################

n=70000
nmax=75
# number of node per graph
ND=75*np.ones((n,1)) 
# node feature matrix
FF=np.zeros((n,nmax,2))
# one-hot coding output matrix 
YY=np.zeros((n,10))
# Convolution kernels, supports
SP=np.zeros((n,nkernel+1,nmax,nmax),dtype=np.float32)


d=train_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 

    FF[i,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i,gtrt]=1

    if recfield==0:
        M=A
    else:
        M=(A+np.eye(nd))
        for j in range(1,recfield):
            M=M.dot(M) 
    M=(M>0)

    deg = A.sum(axis=0) 
    # normalized Laplacian matrix.
    dis=1/np.sqrt(deg)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
    V,U = np.linalg.eigh(nL) 
    V[V<0]=0

    freqcenter=np.linspace(V.min(),V.max(),nkernel)        
    # design convolution supports (aka edge features)         
    for j in range(0,nkernel): 
        SP[i,j ,0:nd,0:nd]=M* (U.dot(np.diag(np.exp(-(dv*(V-freqcenter[j])**2))).dot(U.T)))
    SP[i,nkernel ,0:nd,0:nd]=np.eye(nd)


d=test_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 

    FF[i+60000,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i+60000,gtrt]=1

    if recfield==0:
        M=A
    else:
        M=(A+np.eye(nd))
        for j in range(1,recfield):
            M=M.dot(M) 
    M=(M>0)

    deg = A.sum(axis=0) 
    # normalized Laplacian matrix.
    dis=1/np.sqrt(deg)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
    V,U = np.linalg.eigh(nL) 
    V[V<0]=0

    freqcenter=np.linspace(V.min(),V.max(),nkernel)        
    # design convolution supports (aka edge features)         
    for j in range(0,nkernel): 
        SP[i+60000,j ,0:nd,0:nd]=M* (U.dot(np.diag(np.exp(-(dv*(V-freqcenter[j])**2))).dot(U.T)))
    SP[i+60000,nkernel ,0:nd,0:nd]=np.eye(nd)

np.save('supports',SP)
np.save('feats',FF)
np.save('output',YY)
np.save('nnodes',ND)


