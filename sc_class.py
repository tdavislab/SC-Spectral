"""SC Class And Generators.

This python script implements a simple class for simplicial complexes.
The vertices are stored as a list / range.
Higher order simplices are stored as list of tuples.
The class provides methods to read simplicial complex from and write to a CSV.
The CSV file needs to be in the following format (similar to OFF):
    - First line is a comma separated list of #simplices of each dimension
    from 0 to K (K is the dimension of the simplicial complex)
    - From the second line onward - each line represents a simplex.
    - a simplex is recorded as a comma separated list of vertex IDs.
    - The last entry is the weight of the simplex.
    - List simplicial complexes in increasing order of dimensions.

Example of a simplicial complex with 3 vertices, 3 edges, 1 triangle.
================
sample_sc.txt
================
3, 3, 1
0, 1.0
1, 1.0
2, 1.0
0, 1, 1.0
0, 2, 1.0
1, 2, 1.0
0, 1, 2, 1.0
================

The script also contains two SC generators.
One generates a complete SC, the other generates a dumbbell shaped SC.

The class provides methods to compute the incidence matrix, the adjacency
matrix, and the up-Laplacian of the simplicial complex for a given dimension.

In addition, it provides a way to visualize the dual graph (using networkx)

"""

import numpy as np
from scipy.sparse import coo_matrix, spdiags
import networkx as nx
from itertools import combinations


class Simplicial_Complex:
    """A Class for Simplicial Complexes.

    Attributes:
    -----------

        _dims: Int. Dimension of the SC

        _nk: list of Ints. Number of simplices of each dim from 0 to _dims

        _simplices: List. _simplices[k]: list of all k-dimensional simplices.

        _incidences: List. _incidences[k]: Incidence matrix for dimension k.

        _laplacians: List. _laplacians[k]: up-Laplacian for dimension k.

        _adjacencies: List. _adjacencies[k]: Adjacency matrix for dimension k.

    The incidence, up-Laplacian and adjacency matrices are not computed at
    initialization. They are stored in attribute lists after first computation.

    Methods:
    --------

    add_simplices: Add simplices to the SC instance.

    getWeights: Return a list of weights for simplices of given dim.

    is_even: Compute whether the orientation of a simplex is even or odd.

    incidence: Return incidence matrix for simplices of given dime.

    up_Laplacian: Return up-Laplacian for given dimension.

    adjacency: Return adjacency matrix for simplices of given dim.

    nxGraph: Return a dual graph for given dim (an instance of networkx graph).

    writeToFile: Write SC to a CSV text file.

    readFromFile: Read SC from a CSV text file.


    """

    def __init__(self, dims, csr_flag=True):
        self._dims = dims
        self._nk = [0]*(dims+1)
        self._simplices = [None]*(dims+1)
        self._incidences = [None]*(dims+1)
        self._laplacians = [None]*(dims+1)
        self._adjacencies = [None]*(dims+1)

    def add_simplices(self, dim, simplices):
        """Add simplices of specified dim to the SC instance.

        Inputs:
        -------
            dim: Int
                - Specify dimension of simplices being added.

            simplices: list of tuples
                - First (dim+1) elements of the tuple are vertex IDs (ordered).
                - The last element in the tuple is the weight of the simplex.

        """
        inputs = ['A list of tuples, where: ',
                  'the first (dim+1) elements are vertex IDs (ordered) and ',
                  'the last element is the weight of the simplex.']
        errorMsg = 'Unexpected input. Accepted input is: '
        errorMsg = errorMsg + '\n'.join(inputs)

        if isinstance(simplices, list):
            if any([not(isinstance(s, tuple) and len(s) == (dim+2))
                    for s in simplices]):
                raise(TypeError(errorMsg+'\nTuple length does not match'))
        else:
            raise(TypeError(errorMsg+'\nInput not a list'))

        self._nk[dim] = len(simplices)
        self._simplices[dim] = simplices

    def getWeights(self, dim):
        """Return a list of weights for simplices of given dim.

        Inputs:
        -------
            dim: Int
                - Dimension of simplices for which weights are returned.

        Outputs:
        --------
            w: list of floats
                - list of weights for the simplices of specified dimension.

        """
        *_, w = map(list, zip(*self._simplices[dim]))
        return(w)

    def is_even(self, simplex):
        """Compute whether the orientation of a simplex is even or odd.

        Inputs:
        -------
            simplex: tuple of vertex IDs
                - Should only include vertices of the simplex, not the weight.

        Outputs:
        --------
            flag: Bool
                - True is the orientation is even, false if orientation is odd.

        """
        count = 0
        for i, val in enumerate(simplex, start=1):
            count += sum(val > val2 for val2 in simplex[i:])

        flag = not(count % 2)

        return(flag)

    def incidence(self, dim, weighted=False):
        """Return incidence matrix for simplices of given dimension.

        Inputs:
        -------
            dim: Int
                - Dimension for which to compute the incidence matrix.

            weighted: Bool
                - If False, this is an indicator matrix (values are 0 or 1)
                - If True, values are (signed) weights.

        Outputs:
        --------
            D: scipy.sparse CSC matrix
                - At first computations, stored in _incidences for future use.

        """
        if(dim < 0 or dim >= self._dims):
            ValueError('dim must be between 0 and '+str(self._dims - 1))

        if self._incidences[dim] is None:
            n1, n0 = self._nk[dim+1], self._nk[dim]
            *S1, W1 = map(list, zip(*self._simplices[dim+1]))
            *S0, W0 = map(list, zip(*self._simplices[dim]))
            S1 = list(zip(*S1))
            S0 = list(map(frozenset, zip(*S0)))
            data = []
            rows = []
            cols = []

            for ii in range(n1):
                s1 = S1[ii]
                c_list = [s1[j:] + s1[:j] for j in range(len(s1))]
                faces = [c[:-1] for c in c_list]
                for s0 in faces:
                    idx = S0.index(set(s0))
                    rows.append(ii)
                    cols.append(idx)
                    if (is_even(s1) ^ is_even(s0)):
                        if weighted:
                            data.append(-W1[ii])
                        else:
                            data.append(-1.0)
                    else:
                        if weighted:
                            data.append(W1[ii])
                        else:
                            data.append(1.0)

            D = coo_matrix((data, (rows, cols)), shape=(n1, n0))
            if self._incidences[dim] is None:
                self._incidences[dim] = D.tocsc()

            return(D.tocsc())
        else:
            return(self._incidences[dim])

    def up_Laplacian(self, dim):
        """Return up-Laplacian for given dimension.

        Inputs:
        -------
            dim: Int
                - Dimension for which to compute the up-Laplacian.

        Outputs:
        --------
            L: scipy.sparse CSC matrix
                - At first computations, stored in _laplacians for future use.

        """
        if(dim < 0 or dim >= self._dims):
            ValueError('dim must be between 0 and '+str(self._dims - 1))

        if self._laplacians[dim] is None:
            n = self._nk[dim+1]
            W = spdiags(self.getWeights(dim+1), 0, n, n)
            D = self.incidence(dim)
            L = D.transpose() @ W @ D
            if self._laplacians[dim] is None:
                self._laplacians[dim] = L.tocsc()

            return(L.tocsc())
        else:
            return(self._laplacians[dim])

    def adjacency(self, dim):
        """Return (oriented) adjacency matrix for simplices of given dim.

        Inputs:
        -------
            dim: Int
                - Dimension for which to compute the adjacency matrix.

        Outputs:
        --------
            A: scipy.sparse CSC matrix
                - At first computations, stored in _adjacencies for future use.

        """
        if(dim < 0 or dim >= self._dims):
            ValueError('dim must be between 0 and '+str(self._dims - 1))

        if self._adjacencies[dim] is None:
            A = self.up_Laplacian(dim)
            A.setdiag(0.0)
            if self._adjacencies[dim] is None:
                self._adjacencies[dim] = A.tocsc()

            return(A.tocsc())
        else:
            return(self._adjacencies[dim])

    def nxGraph(self, dim, draw=False, **kwargs):
        """Return an instance of networkx graph (easier for visualization).

        Inputs:
        -------
            dim: Int
                - Dimension for which to construct the dual graph.

            draw: Bool
                - If True, generate a visualization using networkx.draw()

            **kwargs: Argument value pairs
                - Input arguments for networkx.draw() if draw method is True.

        Outputs:
        --------
            nxG: Instance of a networkx graph

        """
        A = self.adjacency(dim)
        nxG = nx.from_scipy_sparse_matrix(np.abs(A))
        if draw:
            nx.draw_networkx(nxG, **kwargs)

        return(nxG)

    def writeToFile(self, fname):
        """Write SC to a text file.

        Inputs:
        -------
            filename: string
                - Writes the SC to the specified CSV text file.

        Outputs:
        --------
            Returns True if the method terminated successfully.

        """
        from os import linesep
        with open(fname, 'w') as fp:
            mystr = ', '.join(map(str, self._nk))
            fp.write(mystr + linesep)

            for ii in range(self._dims + 1):
                ns = self._nk[ii]
                simplices = self._simplices[ii]

                for jj in range(ns):
                    mystr = ', '.join(map(str, simplices[jj]))
                    fp.write(mystr+linesep)

        fp.close()
        return(True)

    def readFromFile(self, fname):
        """Read SC from a file.

        Inputs:
        -------
            filename: string
                - Read the SC from the specified CSV text file.

        Outputs:
        --------
            Returns True if the method terminated successfully.
        """
        from os import linesep
        with open(fname, 'r') as fp:
            l0 = fp.readline().replace(linesep, '').split(',')
            self._dims = len(l0) - 1
            nk = list(map(np.int_, l0))
            for ii in range(len(nk)):
                s_ii = np.asarray(np.genfromtxt(fp, delimiter=',',
                                                skip_header=0,
                                                max_rows=nk[ii]))
                w_ii = s_ii[:, -1]
                s_ii = np.int_(s_ii[:, :-1])
                simplices = list(zip(*list(map(tuple, s_ii.T)), w_ii))
                self.add_simplices(dim=ii, simplices=simplices)

        return(True)


"""SC Generators."""


def generateCompleteSC_2D(nvert):
    """Generate a complete 2D SC with given number of vertices.

    Inputs:
          nvert: Integer
                - number of vertices

    Outputs:
          SC: Instance of the SC class: see sc_class.py
              - Complete SC composed of:
                  all possible edges and triangles between 'nvert' vertices.
    """
    vertexIDs = list(range(nvert))
    s0 = list(zip(vertexIDs, [1.0]*nvert))
    s1 = [s+(1.0,) for s in combinations(vertexIDs, 2)]
    s2 = [s+(1.0,) for s in combinations(vertexIDs, 3)]

    SC = Simplicial_Complex(2)
    SC.add_simplices(0, s0)
    SC.add_simplices(1, s1)
    SC.add_simplices(2, s2)

    return(SC)


def generateDumbbellSC_2D(nvert):
    """Generate a 2D dumbbell-shaped SC with given number of vertices.

    Inputs:
          nvert: Integer
                - number of vertices
    Outputs:
          SC: Instance of SC class. See sc_class.py
              - Dumbbell shaped SC composed of two complete subcomplexes.
              - Each complete subcomplex consists of nvert/2 nodes
              - Two subcomplexes are connected via:
                  # cross edges and
                  # cross triangles
    """
    s0 = list(zip(range(nvert), [1.0]*nvert))

    # Subcomplex 1
    vertexIDs_1 = range(np.int_(nvert/2))
    e1 = [s+(1.0,) for s in combinations(vertexIDs_1, 2)]
    f1 = [s+(1.0,) for s in combinations(vertexIDs_1, 3)]

    # Subcomplex 2
    vertexIDs_2 = range(np.int_(nvert/2), nvert)
    e2 = [s+(1.0,) for s in combinations(vertexIDs_2, 2)]
    f2 = [s+(1.0,) for s in combinations(vertexIDs_2, 3)]

    # edges and triangles across subcomplex 1 and 2
    n2 = np.max([2, int(nvert/5)])
    vertexIDs_3 = np.sort(np.random.choice(vertexIDs_1, n2, replace=False))
    vertexIDs_4 = np.sort(np.random.choice(vertexIDs_2, n2, replace=False))
    l1 = list(combinations(vertexIDs_3, 2))
    l2 = list(combinations(vertexIDs_4, 2))

    cross_e = set()
    cross_f = set()
    for i in l1:
        for j in l2:
            [cross_e.add((p, q)) for p in i for q in j]
            cross_f.add(i+(j[0],))
            cross_f.add(i+(j[1],))
            cross_f.add((i[0],)+j)
            cross_f.add((i[1],)+j)
    e3 = [s+(1.0,) for s in cross_e]
    f3 = [s+(1.0,) for s in cross_f]

    SC = Simplicial_Complex(2)
    SC.add_simplices(0, s0)
    SC.add_simplices(1, e1+e2+e3)
    SC.add_simplices(2, f1+f2+f3)

    return(SC)


"""Utility methods."""


def is_even(simplex):
    """
    Compute whether the orientation of a simplex is even or odd.

    We determine whether the vertex ID sequence in the tuple is even or odd
    by counting the number of inversions.

    Parameters
    ----------
    sequence : a tuple of Ints
        a tuple of vertex IDs

    Returns
    -------
    sgn: Bool
        Sign of the given sequence of vertex IDs.
        (parity or sign of the permutation - it is either even or odd)

        False or 0 => odd sequence
         True or 1 => even sequence

    """
    count = 0
    for i, val in enumerate(simplex, start=1):
        count += sum(val > val2 for val2 in simplex[i:])

    return(not(count % 2))
