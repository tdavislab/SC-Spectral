"""Spectral algorithms for simplicial complexes."""
from sc_class import np, Simplicial_Complex
from scipy.sparse import spdiags, eye, csc_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def sparsify(SC, dim, q, iseed=12345678):
    """Sparsify input simplicial complex at the given dimension.

    Parameters
    ----------
    SC : TYPE
        DESCRIPTION.
    dim : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.
    iseed : TYPE, optional
        DESCRIPTION. The default is 12345678.

    Returns
    -------
    None.

    """
    m = SC._nk[dim]
    *F, W = map(list, zip(*SC._simplices[dim]))
    F = list(zip(*F))
    D = SC.incidence(dim-1)
    invL = np.linalg.pinv(SC.up_Laplacian(dim-1).todense())
    R = D @ invL @ D.transpose()
    denom = R.diagonal().A1.dot(W)
    pdf = [W[i]*R[i, i]/denom for i in range(m)]

    F_sampled = list(np.random.choice(range(m), q, pdf))
    hst = {z: F_sampled.count(z) for z in F_sampled}

    spSC = Simplicial_Complex(SC._dims)
    for ii in range(dim):
        spSC._nk[ii] = SC._nk[ii]
        spSC._simplices[ii] = SC._simplices[ii]

    for ii in range(dim-1):
        spSC._incidences[ii] = spSC._incidences[ii]
        spSC._laplacians[ii] = SC._laplacians[ii]
        spSC._adjacencies[ii] = SC._adjacencies[ii]

    spF = []
    for idx in hst.keys():
        w = W[idx] * hst[idx] / q * pdf[idx]
        spF.append(F[idx] + (w,))

    spSC._nk[dim] = len(spF)
    spSC._simplices[dim] = spF

    return(spSC)


def cluster_NJW(SC, dim, num_clust, num_jobs=1):
    """Spectral clustering for simplicial complex at specified dimension.

    Parameters
    ----------
    SC : TYPE
        DESCRIPTION.
    dim : TYPE
        DESCRIPTION.
    num_clust : TYPE
        DESCRIPTION.
    num_jobs : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    A = np.abs(SC.adjacency(dim))
    m = A.shape[0]
    D = spdiags(np.power(A.sum(axis=0).A1, -1./2), 0, m, m, format='csc')
    M = D @ A @ D
    val, vec = eigsh(M, k=num_clust)
    pset = normalize(vec)
    model = KMeans(n_clusters=num_clust, n_jobs=num_jobs)
    model.fit(pset)
    labels = model.fit_predict(pset)

    return(labels)


def labelPropagate_dualGraph(SC, dim, labels):
    """Label propagation for simplicial complex at given dimension.

    Parameters
    ----------
    SC : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    A = np.abs(SC.adjacency(dim))
    m = A.shape[0]
    D = spdiags(np.power(A.sum(axis=0).A1, -1), 0, m, m, format='csc')
    P = D @ A

    idx_known = np.where(np.asarray(labels) != 0)[0]
    yl = [labels[idx] for idx in idx_known]
    idx_unknown = np.where(np.asarray(labels) == 0)[0]

    Puu = P[idx_unknown, :][:, idx_unknown]
    Pul = P[idx_unknown, :][:, idx_known]
    T1 = (eye(Puu.shape[0], format='csc')-Puu).todense()
    temp = csc_matrix(np.linalg.pinv(T1))
    yu = temp.dot(Pul.dot(yl))
    for j in range(len(yu)):
        labels[idx_unknown[j]] = yu[j]

    labels = np.sign(labels)

    return(labels)
