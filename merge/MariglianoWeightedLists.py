from __future__ import division
import numpy as np


class MariglianoWeightedLists:
    def __init__(self, Z, L):
        """
        This algorithm will use a linkage matrix Z to compute weights for the lists used by the matrix Z.
        :param Z: the linkage matrix, Warning: 3x3 shaped not 3x4 like scipy returns. Drop the last column if needed.
        :param L: the number of lists used in Z
        """
        self._Z = Z
        self._L = L

    def compute_weights(self):
        tree, z_weights = self._build_tree()
        W = np.zeros(self._L)
        self._update_weights(tree, z_weights, W, max_balance=1.0, parent_node_weight=z_weights[tree.id])

        assert abs(sum(W) - 1.0) < 1e-3
        return W

    def _build_tree(self):
        """
        Build a tree from the linkage matrix Z.
        :return: the tree
        """
        roots = []
        z_weights = {}

        for i, z_i in enumerate(self._Z):
            idx = self._L + i
            lc = z_i[0]  # type: Node
            rc = z_i[1]  # type: Node
            roots.append((Node(idx, lc, rc)))
            z_weights[idx] = z_i[2]

        root = self._find_max_node(roots)
        tree = self._replace_children(root, roots)

        return tree, z_weights

    @staticmethod
    def _find_max_node(roots):
        """
        Find the max root among all the roots (as "roots" contains multiple small trees of 3 nodes)
        Example 3 -> (2,1) and 4 -> (3,0) will returns 4 -> (3,0)
        :param roots:
        :return: the max root which will be used as the top level node (root) for the tree.
        """
        return max(roots, key=lambda x: x.id)

    def _replace_children(self, node, roots):
        """
        Replace the children of a node (which are integers at the beginning) with Node objects.
        :param node: the node whose children will be replaced
        :param roots: in order to retrieve the nodes objects from an integer
        :return: the node with its children replaced by Node objects
        """
        if isinstance(node, float) and node not in [r.id for r in roots]:
            return Node(node, 0, 0)

        if isinstance(node, float) and node in [r.id for r in roots]:
            id = node
            r = [r for r in roots if r.id == id][0]
            lc = r.lc
            rc = r.rc
            return Node(id, self._replace_children(lc, roots), self._replace_children(rc, roots))

        if isinstance(node, Node):
            node.lc = self._replace_children(node.lc, roots)
            node.rc = self._replace_children(node.rc, roots)

        return node

    def _update_weights(self, node, z_weights, W, max_balance, parent_node_weight):
        """
        Update the weights recursively by distributing the max_balance
        :param node: the node to update
        :param z_weights: a dictionary that contains the distances from the matrix
        :param W: the weights
        :param max_balance: the "amount"/ratio of weight we can use for this node (and its direct children)
        :param parent_node_weight: the weight of the parent node
        :return: W, the updated weights as a list
        """
        # case 1: if both left child and right child are node
        if not node.lc.is_leaf() and not node.rc.is_leaf():
            lc = node.lc
            rc = node.rc

            # instead of computing a ratio, we compute the difference between the nodes
            r = abs(z_weights[lc.id] - z_weights[rc.id])
            rs = self._rescale(r, max_balance, min_ratio=0.5)

            if z_weights[lc.id] > z_weights[rc.id]:
                z_weights[lc.id] = rs
                z_weights[rc.id] = max_balance - rs
            else:
                z_weights[lc.id] = max_balance - rs
                z_weights[rc.id] = rs

            self._update_weights(lc, z_weights, W, max_balance=z_weights[lc.id], parent_node_weight=z_weights[node.id])
            self._update_weights(rc, z_weights, W, max_balance=z_weights[rc.id], parent_node_weight=z_weights[node.id])

        # case 2: if left or right child is a node
        if (not node.lc.is_leaf() and node.rc.is_leaf()) or (node.lc.is_leaf() and not node.rc.is_leaf()):
            lc = node.lc
            rc = node.rc

            # retrieve the child node and the leaf node
            child_node = lc if not lc.is_leaf() else rc
            leaf_node = lc if lc.is_leaf() else rc

            # the ratio is computed between the parent node and the child node
            r = z_weights[child_node.id] / float(parent_node_weight)

            rs = self._rescale(r, max_balance, min_ratio=1.0 / 3.0)

            # limit the weight to 2/3 if the ratio is too big. For example: 0.8/0.9 -> yield big ratio but lists are
            # very different
            rs = min(rs, 2.0 / 3.0)
            z_weights[child_node.id] = rs
            W[leaf_node.id] = max_balance - rs

            # update the child node
            self._update_weights(child_node, z_weights, W, max_balance=z_weights[child_node.id],
                                 parent_node_weight=z_weights[node.id])

        # case 3: both left and right are leaves
        if node.lc.is_leaf() and node.rc.is_leaf():
            lc = node.lc
            rc = node.rc

            # As they are no more nodes, the balance is split among the leaves
            W[lc.id] = 0.5 * max_balance
            W[rc.id] = 0.5 * max_balance

    @staticmethod
    def _rescale(r, max_balance, min_ratio):
        """
        Re-scale the ratio of two nodes to be in [min_ratio*max_balance,max_balance].
        :param r: ratio
        :param max_balance: the maximum weight balance to use. This max_balance is equals to 1 for the root
        and then it is split between the nodes as it goes down the tree.
        :return: the re-scaled ratio
        """
        return r * (max_balance - min_ratio * max_balance) + min_ratio * max_balance


class Node:
    def __init__(self, id, lc, rc):
        """
        Node is a element a of tree. It can be the root, a node or a leaf.
        It contains an id a left child (lc) and a right child (rc).
        If lc and rc are set to 0 then the node is a leaf. So lc and rc can both either a Node object or a int
        :param id: the node id i.e. the id of the list
        :param lc: the left child
        :param rc: the right child
        """
        self.id = int(id)
        self.lc = lc
        self.rc = rc

    def is_leaf(self):
        return self.lc == 0 and self.rc == 0

    def __repr__(self):
        if self.is_leaf():
            return "[%s]" % self.id
        else:
            return "[%s -> (%s, %s)]" % (self.id, self.lc, self.rc)


if __name__ == '__main__':
    from matplotlib import pyplot as plt


    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d

        from scipy.cluster.hierarchy import dendrogram
        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata


    examples = []

    # # Demo matrix - all cases
    # L = 5
    # Z = [[0, 2, 0.1, 1],
    #      [4, 3, 0.3, 1],
    #      [1, 5, 0.4, 1],
    #      [6, 7, 0.8, 1]]
    # examples.append((L, Z, [1, 1, 1, 1, 1]))
    #
    # L = 3
    # Z = [[2, 1, 0, 1],
    #      [0, 3, 0.6, 1]]
    # examples.append((L, Z, [1, 1, 1]))
    #
    # # right node near 0
    # L = 3
    # Z = [[0, 2, 0.1, 1],
    #      [1, 3, 0.9, 1]]
    # examples.append((L, Z, [0.1, 0.8, 0.1]))
    #
    # # left node near 0
    # L = 3
    # Z = [[0, 1, 0.1, 1],
    #      [2, 3, 0.9, 1]]
    # examples.append((L, Z, [0.1, 0.1, 0.8]))
    #
    # # 1/3 for all
    # L = 3
    # Z = [[0, 2, 0.8, 1],
    #      [1, 3, 0.9, 1]]
    # examples.append((L, Z, [0.33, 0.33, 0.33]))
    #
    # # Neves example 1
    # L = 3
    # Z = [[0, 2, 0.25, 1],
    #      [1, 3, 0.8, 1]]
    # examples.append((L, Z, [0.275, 0.45, 0.275]))
    #
    # # Neves examples 2
    # L = 4
    # Z = [[1, 3, 0.25, 1],
    #      [0, 2, 0.8, 1],
    #      [4, 5, 1, 1]]
    # examples.append((L, Z, [0.31, 0.19, 0.31, 0.19]))
    #
    # # 4 all equal
    # L = 4
    # Z = [[1, 3, 0.2, 1],
    #      [0, 2, 0.2, 1],
    #      [4, 5, 0.8, 1]]
    # examples.append((L, Z, [0.25, 0.25, 0.25, 0.25]))

    # A1
    L = 3
    Z = [[0, 2, 0.05, 1],
         [1, 3, 0.8, 1]]
    examples.append((L, Z, [0.15, 0.7, 0.15]))

    # B1
    L = 3
    Z = [[0, 2, 0.25, 1],
         [1, 3, 0.8, 1]]
    examples.append((L, Z, [0.275, 0.45, 0.275]))

    # C1
    L = 3
    Z = [[0, 2, 0.7, 1],
         [1, 3, 0.8, 1]]
    examples.append((L, Z, [0.3, 0.4, 0.3]))

    # A2
    L = 4
    Z = [[1, 3, 0.25, 1],
         [0, 2, 0.8, 1],
         [4, 5, 1.0, 1]]
    examples.append((L, Z, [0.31, 0.19, 0.31, 0.19]))

    # B2
    L = 4
    Z = [[1, 3, 0.5, 1],
         [0, 2, 0.7, 1],
         [4, 5, 0.8, 1]]
    examples.append((L, Z, [0.3, 0.2, 0.3, 0.2]))

    # C2
    L = 4
    Z = [[1, 3, 0.2, 1],
         [0, 2, 0.2, 1],
         [4, 5, 0.8, 1]]
    examples.append((L, Z, [0.25, 0.25, 0.25, 0.25]))

    # for e in [examples[4]]:
    i = 0
    for e in examples:
        L, Z, W = e

        fancy_dendrogram(Z)

        Z = np.array(Z)
        Z = Z[:, :-1]  # get rid of the last column which is not used

        mwl = MariglianoWeightedLists(Z, L)
        mwl.compute_weights()

        print("W expected", W)
        print("")
        print("")

        plt.show()
