class MariglianoWeightLists:
    def __init__(self, Z, L):
        self._Z = Z
        self._L = L

    def compute_weights(self):
        tree, z_weights = self._build_tree()
        W = np.zeros(self._L)
        self._update_weights(tree, z_weights, W, max_balance=1.0, parent_node_weight=z_weights[tree.id])

        # reverse the weights since we want to promote the diversity
        s = sum([1.0 / w for w in W])
        print("W", W)
        W = map(lambda w: (1.0 / w) / s, W)
        print("W", W)
        print("W sum", sum(W))
        return W

    def _build_tree(self):
        roots = []
        z_weights = {}

        for i, z_i in enumerate(self._Z):
            print(i, z_i)
            idx = self._L + i
            lc = z_i[0]  # type: Node
            rc = z_i[1]  # type: Node
            roots.append((Node(idx, lc, rc)))
            z_weights[idx] = z_i[2]

        print(roots)

        root = self._find_max_node(roots)
        print("root", root)
        tree = self._replace_children(root, roots)
        print("tree", tree)

        print("z weigths", z_weights)

        return tree, z_weights

    @staticmethod
    def _find_max_node(roots):
        return max(roots, key=lambda x: x.id)

    def _replace_children(self, node, roots):
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

       :type node: Node
       """
        print(" ")
        print(" ")
        print("current node is", node.id)

        # case 1: if both left child and right child are node
        if not node.lc.is_leaf() and not node.rc.is_leaf():
            print("case 1")
            lc = node.lc
            rc = node.rc
            print(lc.id, rc.id)

            r = abs(z_weights[lc.id] - z_weights[rc.id])
            rs = self._rescale(r, max_balance)

            if z_weights[lc.id] > z_weights[rc.id]:
                z_weights[lc.id] = rs
                z_weights[rc.id] = max_balance - rs
            else:
                z_weights[rc.id] = rs
                z_weights[lc.id] = max_balance - rs

            self._update_weights(lc, z_weights, W, max_balance=z_weights[lc.id], parent_node_weight=z_weights[node.id])
            self._update_weights(rc, z_weights, W, max_balance=z_weights[rc.id], parent_node_weight=z_weights[node.id])

        # case 2: if left or right child is a node
        if (not node.lc.is_leaf() and node.rc.is_leaf()) or (node.lc.is_leaf() and not node.rc.is_leaf()):
            print("case 2")
            lc = node.lc
            rc = node.rc
            print(lc.id, rc.id)

            child_node = lc if not lc.is_leaf() else rc
            leaf_node = lc if lc.is_leaf() else rc

            r = z_weights[child_node.id] / float(parent_node_weight)
            rs = self._rescale(r, max_balance)

            z_weights[child_node.id] = rs
            W[leaf_node.id] = max_balance - rs

            self._update_weights(child_node, z_weights, W, max_balance=z_weights[child_node.id],
                                 parent_node_weight=z_weights[node.id])

        # case 3: both left and right are leaves
        if node.lc.is_leaf() and node.rc.is_leaf():
            print("case 3")
            lc = node.lc
            rc = node.rc
            print(lc.id, rc.id)
            W[lc.id] = 0.5 * max_balance
            W[rc.id] = 0.5 * max_balance

    @staticmethod
    def _rescale(r, max_balance):
        return r * (max_balance - 0.5 * max_balance) + 0.5 * max_balance


class Node:
    def __init__(self, id, lc, rc):
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
    import numpy as np
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


    # Demo matrix - all cases
    L = 5
    Z = [[0, 2, 0.1, 1],
         [4, 3, 0.3, 1],
         [1, 5, 0.4, 1],
         [6, 7, 0.8, 1]]

    # # Neves example 1
    # L = 3
    # Z = [[0, 2, 0.25, 1],
    #      [1, 3, 0.8, 1]]

    # # Neves examples 2
    # L = 4
    # Z = [[1, 3, 0.25, 1],
    #      [0, 2, 0.8, 1],
    #      [4, 5, 1, 1]]

    fancy_dendrogram(Z)
    plt.show()

    Z = np.array(Z)
    Z = Z[:, :-1]

    mwl = MariglianoWeightLists(Z, L)
    mwl.compute_weights()
