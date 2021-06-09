
import numpy as np
import networkx as nx
import scipy.io as sio
dataset = nx.read_graph6('dataset/graph8c/raw/graph8c.g6')

Ao=np.zeros((len(dataset),8,8))
for i in range(0,len(dataset)):
    Ao[i,:,:] = nx.to_numpy_matrix(dataset[i])

sio.savemat('graph8c',{'A':Ao})




a=1