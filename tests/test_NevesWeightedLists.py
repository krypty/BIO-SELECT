import pytest

from merge.NevesWeightedLists import NevesWeightedLists
import numpy as np


class TestNevesWeightedLists:
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_paper_example1(self):
        """
        This test is inspired from the example 1 of the paper "Ensemble combination rules" by Aitana Neves
        """
        Z = np.array([
            [0, 2, 0.25],
            [1, 3, 0.8]
        ])

        L = 3

        W = NevesWeightedLists.compute_weights(Z, L)
        print(W)

        W_synthetic = np.array([0.275, 0.45, 0.275])
        assert TestNevesWeightedLists._is_equals(W, W_synthetic)

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_paper_example2(self):
        """
        This test is inspired from the example 2 of the paper "Ensemble combination rules" by Aitana Neves
        """
        Z = np.array([
            [0, 2, 0.25],
            [1, 3, 0.8],
            [4, 5, 1]
        ])

        L = 4

        W = NevesWeightedLists.compute_weights(Z, L)
        print(W)

        W_synthetic = np.array([0.19, 0.31, 0.19, 0.31])
        assert TestNevesWeightedLists._is_equals(W, W_synthetic)

    @staticmethod
    def _is_equals(W, W_synthetic):
        for a, b in zip(W, W_synthetic):
            epsilon = 0.01  # big epsilon in order to match the approximated results from the paper
            assert abs(a - b) < epsilon, "%s and %s are not equal" % (a, b)
        return True
