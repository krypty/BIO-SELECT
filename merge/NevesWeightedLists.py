from __future__ import division
import numpy as np


class NevesWeightedLists:
    """
    This class implements the algorithm proposed by Aitana Neves (HEIG-VD, 2015) in a paper called "Ensemble combination
    rules". I do not own/claim any rights of this algorithm. This class is provided as is without warranty.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_weights(Z, L):
        """

        :param Z: linkage matrix describing the hierarchical clustering distance of a distance matrix D.
        This matrix is an (L-1) x 3 matrix containing cluster tree information. The first two columns $Z_i,1 < Z_i,2$
        refer to indices of the classifiers forming a cluster. The third column contains the linkage distance
        between the objects being paired.
        Warning: if you use the linkage method from scipy (scipy.cluster.hierarchy.linkage) you will get a (L-1) x 4
        matrix. As the documentation says : "The fourth value Z[i, 3] represents the number of original observations
        in the newly formed cluster.". You need to get rid of it in order to use this class
        :return: array of weights of the lists
        """

        """
        Notes about Python notations: In the paper the linkage matrix contains both integers and real numbers.
        In Python, we use a matrix with real numbers, so we must cast the indices in order to retrieve the indices
        of the W matrix.
        Thus, both matrices and arrays start at 0, so we need to be careful with the mathematical notation.
        """

        # init the weights to 0
        W = np.zeros(L)

        # for each row in the linkage matrix
        for i in range(L - 1):
            if Z[i, 0] < L and Z[i, 1] < L:
                W[int(Z[i, 0])] = 0.5
                W[int(Z[i, 1])] = 0.5

            if Z[i, 0] < L and Z[i, 1] >= L:
                idx = int(Z[i, 1] % L)  # find the corresponding row in matrix Z
                rs = (Z[idx, 2] / Z[i, 2]) * (2 / 3 - 1 / 2) + 1 / 2  # re-scale from [0;1] to [1/2;2/3]
                W[int(Z[i, 0])] = 1 - rs
                W[int(Z[idx, 0])] *= rs
                W[int(Z[idx, 1])] *= rs

            if Z[i, 0] >= L and Z[i, 1] >= L:
                idx0 = int(Z[i, 0] % L)
                idx1 = int(Z[i, 1] % L)
                rs = (Z[idx0, 2] / Z[idx1, 2]) * (1 / 2 - 1 / 3) + 1 / 3  # re-scale from [0;1] to [1/3;1/2]
                W[int(Z[idx0, 0])] *= rs
                W[int(Z[idx0, 1])] *= rs
                W[int(Z[idx1, 0])] *= (1 - rs)
                W[int(Z[idx1, 1])] *= (1 - rs)

        return W
