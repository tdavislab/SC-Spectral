# Spectral Algorithms for Simplicial Complexes

This repository implements spectral sparsification, spectral clustering, and label propagation algorithms for simlicial complexes.<br>
These algorithms are based on the paper [Spectral sparsification of simplicial complexes for clustering and label propagation](https://journals.carleton.ca/jocg/index.php/jocg/article/view/417).

The main code is divided into two files - `sc_class.py` and `sc_algorithms.py`.

`sc_class.py`: provides a simple *Simplicial_Complex* class that simplifies interaction with simplicial complexes. It also proved two simple generators - one to generate a complete 2D complex (all possible edges and triangles for specified number of vertices) and the other to generate a dumbbell-shaped 2D complex (two complete sub-complexes connected by cross edges and triangles).

`sc_algorithms.py`: provides algorithms for spectral sparsification, spectral clustering and label propagation on simplicial complexes at a fixed dimension.

The notebooks - `Examples_SC_Class_Usage.ipynb` and `Examples_SC_Spectral_Algorithms.ipynb` illustrate how to use the *Simplicial_Complex* class and apply the spectral algorithms to simplicial complexes.

The *Simplicial_Complex* class provides methods to read a simplicial complex from a text file. The text file must specify simplicial complex in a particular format. The required format is described in the `Examples_SC_Class_Usage.ipynb` notebook. The directory `sc_examples` has examples of toy simplicial complexes.

Users should note that

a) The *Simplicial_Complex* class allows users to easily generate simplicial complexes or read them from a text file. However, it does not perform input validation, i.e., it is assumed that the input simplices form a valid simplicial complex.

b) In our spectral sparsification examples, we assume that the sparsification is done at the top dimension. The algorithm can be used to sparsify the simplicial complex at an intemediate dimension `k`. But for every `k-simplex` removed by the sparsifier, all its higher dimensional co-faces need to be identified and removed from the simplicial complex. This last step is not implemented in the current algorithm.

c) For a simplicial complex of dimension K, spectral clustering and label propagation can be performed at any dimension k, where 0 <= k < K. Since we use the adjacency between k-simplices through shared (k+1) dimensional co-faces, these algorithms cannot be applied at the top dimension.

<hr>

### Requirements
The code is written for Python 3.6 and has been tested with the following package configuration:

```
numpy==1.18.1
scikit-learn==0.22.1      # Only KMeans from sklearn.cluster and normalize from sklearn.preprocessing is used.
scipy==1.4.1              # We use sparse matrices from scipy.sparse and scipy.sparse.linalg
networkx==2.4             # Only used for drawing dual graphs
matplotlib==3.1.3
```

I have used the `@` operator for matrix multiplication in several places which is only defined in python 3.x<br>
For the code to work in python 2.x, this needs to be replaced with `numpy.dot`<br>
The rest of the code should work just the same for python 2.x and/or with older versions of the packages (although I haven't tested it)

<hr>

If you find this code useful, please cite our paper:

```
@Article{OstingPalandeWang2020,
	author   = {Osting, Braxton and Palande, Sourabh and Wang, Bei},
	title    = {Spectral sparsification of simplicial complexes for clustering and label propagation.},
	year     = {2020},
	journal  = {Journal of Computational Geometry ({JoCG})},
	volume   = {11},
	number   = {1},
	pages    = {176--211}
}
```
