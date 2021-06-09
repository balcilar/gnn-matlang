# Breaking the Limits of Message Passing Graph Neural Networks

This repository consists of codes of ["Breaking the Limits of Message Passing Graph Neural Networks"](https://arxiv.org/pdf/2106.04319.pdf) paper published in ICML2021.

## Requirements
These libraries' versions are not strictly needed. But these are the configurations in our test machine. Also, all dependencies of pytorch-geometric are needed.
- Python==3.8.5
- pytorch==1.5.1
- pytorch_geometric==1.6.1
- numpy==1.19.1
- scipy==1.5.2
- networkx==2.5.1
- matplotlib==3.3.1

For tensorflow codes our test environment has following libraries' versions
- Python==3.6.5
- tensorflow-gpu==1.15.0
- numpy==1.17.4
- matplotlib==3.1.2
- scipy==1.3.1
- networkx==2.4
- pickle==4.0

# Reproducing Results

In order to easy debugging and more readabilty of the codes, we hardcoded hyperparameters of each model and also selection of model out of MLP, GCN, GAT, GIN, ChebNet, PPGN and our proposals named GNNML1 and GNNML3 in each problem set's concerned script file separately. Desired model and its hyperparameter's can be changed according to your selection. Note all provided scripts (except isomorphism test codes) are for single run. To get given statistics in the paper, they should be run many times under different random seed.

### Table-1

graph8c.py, sr25.py, exp_iso.py and exp_classify.py is for each column's of Table1 respectively.

### Table-2
counting.py is for result of Table2. For each column, *ntask* variable in Line 381, can be set out of 0,1,2,3,4 accordingly. 

### Table-3
For first 3 column of Table3, filtering.py script can be run with desired task by selection of *ntask* in Line 302 out of 0,1 or 2. Last colmun's result can be taken using freqclass.py script. 

### Table-4
For Zinc dataset result, Zinc12k.py script is needed for all methods. For Mnist75 result mnist75.py script is needed all methods except GNML3. For GNNML3, first you should run prepareMnist_gnnml3_tf.py for preprocessing step. It saves the preprocessed data into file. Then mnist75_gnnml3_tf.py should be run.

### Table-5
Each column can be obtained by mutag.py, enzymes.py, enzymes_contfeat.py, proteins.py and ptc.py respectively. Just GNNML3 result for 3rd column can also be obtained by tensorflow implementation by using enzymes_contfeats_gnnml3_tf.py 

## Citation
Please cite this paper if you use codes and/or datasets in your work,

	@inproceedings{
	balcilar2021breaking,
	title={Breaking the Limits of Message Passing Graph Neural Networks},
	author={Muhammet Balcilar and Pierre H{\'e}roux and Benoit Ga{\"u}z{\`e}re and Pascal Vasseur and S{\'e}bastien Adam and Paul Honeine},
	booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
	year={2021}	
	}

  
## License
MIT License
