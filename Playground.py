from __future__ import division
from time import sleep

from scipy.cluster.hierarchy import ClusterNode
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from datasets.Golub99.GolubDataset import GolubDataset
from datasets.MILE.MileDataset import MileDataset


def main():
    # ds = MileDataset()
    ds = GolubDataset()

    X = ds.get_X()
    y = ds.get_y()
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    classifier = OneVsRestClassifier(ExtraTreesClassifier(n_jobs=-1, n_estimators=99), n_jobs=-1)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))
    yolo = classifier.estimators_[0].feature_importances_
    print(len(yolo))
    print(sorted(yolo, reverse=True)[:30])

    # clf = fs.SelectKBest(fs.f_classif, k=300)  # k is number of features.
    # clf.fit(X_train, y_train)
    #
    # print(clf.scores_[:10])

    # print(len(X_train[1]))
    # X_train_selected = classifier.fit_transform(X_train, y_train)
    # print(len(X_train_selected[1]))


def main2():
    ds = MileDataset()
    # ds = GolubDataset()

    X = ds.get_X()
    y = ds.get_y()
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = Pipeline([('f_classif', SelectKBest(f_classif, k=1000)),
                    ('svm', OneVsRestClassifier(LinearSVC()))])

    clf.fit(X_train, y_train)

    # X_train_selected = clf.fit_transform(X_train, y_train)
    # print(len(X_train_selected[1]))

    predictions = clf.predict(X_train)
    print(predictions)

    score = clf.score(X_test, y_test)
    print(score)

    s = clf.named_steps["f_classif"].scores_
    print(s[:10])

    support = clf.named_steps['f_classif'].get_support()

    features = enumerate(support)
    used_features = [f[0] for f in features if f[1] == True]
    print("Nb used features : %d " % len(used_features))
    print("5 used features : %s" % used_features[:5])


def main3():
    ds = GolubDataset()
    import numpy as np
    from numpy.lib.format import open_memmap

    size = int(1e7)
    # # make some test data
    # data = np.random.randn(size, 2)
    # np.savetxt('/tmp/data_test.csv', data, delimiter=',')
    #
    # sys.exit(0)

    print("Loading train...")
    # we need to specify the shape and dtype in advance, but it would be cheap to
    # allocate an array with more rows than required since memmap files are sparse.
    # mmap_train = open_memmap('/tmp/arr.npy', mode='w+', dtype=np.double, shape=(size, 2))

    mmap_train = open_memmap('/tmp/arr.npy', mode='r', dtype=np.double, shape=(size, 2))

    # # parse at most 10000 rows at a time, write them to the memmaped array
    # n = 0
    # for chunk in pd.read_csv('/tmp/data_train.csv', chunksize=10000):
    #     mmap_train[n:n + chunk.shape[0]] = chunk.values
    #     n += chunk.shape[0]

    print(type(mmap_train))

    X_train = mmap_train[:, 0].reshape(-1, 1)
    print(X_train)
    y_train = mmap_train[:, 1]
    print(y_train)

    print("Loading test...")
    # we need to specify the shape and dtype in advance, but it would be cheap to
    # allocate an array with more rows than required since memmap files are sparse.
    # mmap_test = open_memmap('/tmp/arr_test.npy', mode='w+', dtype=np.double, shape=(size, 2))
    mmap_test = open_memmap('/tmp/arr_test.npy', mode='r', dtype=np.double, shape=(size, 2))

    # # parsest))

    X_test = mmap_test[:, 0].reshape(-1, 1)
    print(X_test)
    y_test = mmap_test[:, 1]
    print(y_test)

    print("Fit will start in 3 sec")
    sleep(3)

    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    knn.fit(X_train, y_train)
    mmap_train.flush()

    print("Score will start in 10 sec")
    sleep(10)
    print("Scoring...")

    score = knn.score(X_test, y_test)
    print("score", score)
    mmap_test.flush()


    # print(np.allclose(data, mmap))
    # True


def main4():
    """
    Parse the limma feature list
    :return:
    """

    import pandas as pd

    ds = MileDataset(samples_limit=20)

    filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/limma-mile-final.csv'
    df = pd.read_csv(filename, sep="\t", usecols=["ID"])
    limma_features_names = df["ID"].values

    feat_ids = ds.get_features_indices(limma_features_names)

    for idx in feat_ids:
        print("%s,%s" % (idx, ds.get_features_names([idx])))
    print(len(feat_ids))  # should be 1000


def main5():
    DELIMITER = ";"

    def import_features():
        # filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/code/outputs/features_lists_final/MILE_determinist2_features_by_rank.csv'
        # filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/code/outputs/features_lists_final/mile_features_by_rank.csv'
        # filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/code/outputs/features_lists_final/golub_19122016_features_by_rank.csv'
        # filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/code/outputs/features_lists_final/golub_limma_features_by_rank.csv'
        filename = r'/home/gary/Dropbox/Master/3emeSemestre/TM/code/outputs/features_lists_final/mile_limma_features_by_rank.csv'

        list_of_features = {}
        with open(filename, "rb") as csvfile:
            for line in csvfile:
                line_splited = line.split(DELIMITER)

                # ignore the algorithm if it does not provide any list
                if len(line_splited) <= 2:
                    continue

                alg_name, feats = line_splited[0], line_splited[1:]

                feats = [(int(f[0]), float(f[1])) for f in [feat.split(",") for feat in feats]]
                list_of_features[alg_name] = feats
        return list_of_features

    feats = import_features()
    print(feats)

    def print_converted_rank(alg_name, tuples):
        indices, ranks = zip(*tuples)
        ranks = map(lambda x: 1.0 / (1.0 + x), ranks)
        tuples = zip(indices, ranks)

        # print
        line = alg_name + DELIMITER + DELIMITER.join(["%d,%.5f" % (f[0], f[1]) for f in tuples])
        print(line)

    print_converted_rank("Limma", feats["Limma"])
    # print_converted_rank("F Value", feats["F Value"])
    # print_converted_rank("Fisher Score", feats["Fisher Score"])
    # print_converted_rank("MRMR", feats["MRMR"])


def main6():
    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    def get_mask_of_features(a, union):
        return [u_i in a for u_i in union]

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d

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

    # 0 and (1 and 2) are quite similar
    # 1 and 2 are identical
    # 3 shares half of the features with (1 and 2)
    # 4 is completely different
    L = [[1673, 4376, 6040, 1881, 4643, 2334, 1828, 6224, 4081, 4846],
         [3232, 3222, 3212, 3243, 2356, 2321, 2222, 4643, 2334, 1828],
         [929, 4376, 6224, 22, 4846, 2321, 2222, 4643, 2334, 1828],
         [929, 4376, 6224, 22, 4846, 2321, 2222, 4643, 2334, 1828],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

    L_labels = ["List %d" % i for i in range(len(L))]
    print(L_labels)

    # warning: we lost the sorting with the union

    u = list(reduce(lambda a, b: set(a).union(set(b)), L))
    print(u, "len: ", len(u))

    L_mask = [get_mask_of_features(f, u) for f in L]
    L_mask = np.array(L_mask)
    Z = linkage(L_mask, metric='dice')

    # calculate full dendrogram
    plt.figure(figsize=(12, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    # Z = [[0, 2, 0.2, 2],
    #      [5, 1, 0.6, 3],
    #      [6, 3, 0.8, 4],
    #      [7, 4, 0.9, 10]]

    # Z = [[0, 2, 0.1, 1],
    #      [4, 3, 0.3, 1],
    #      [1, 5, 0.4, 1],
    #      [6, 7, 0.8, 1]]

    Z = np.array(Z)

    d = fancy_dendrogram(
        Z,
        labels=L_labels,
        leaf_rotation=90.,  # rotates the x axis labels,
    )

    # print("iccord", d['icoord'])
    # print("dcoord", d['dcoord'])
    # print("ivl", d['ivl'])
    # print("leaves", d['leaves'])

    def do(node):
        """

        :type node: ClusterNode
        """
        if not node.is_leaf():
            print("%s and %s" % (node.get_left().get_id(), node.get_right().get_id()))
        else:
            print("not a leaf")
        return node

    print(Z)

    plt.show()

    # we must ignore the fourth column to match the expected matrix's shape for NevesWeightedLists
    Z_fixed = Z[:, :-1]
    print("Z fixed : ")
    print(Z_fixed)
    n_lists = len(L)
    print("n lists", n_lists)

    # from merge.NevesWeightedLists import NevesWeightedLists
    # W = NevesWeightedLists.compute_weights(Z_fixed, n_lists)
    # print(W)

    from merge.MariglianoWeightedLists import MariglianoWeightedLists
    mwl = MariglianoWeightedLists(Z, n_lists)
    W = mwl.compute_weights()
    print(W)


def main7():
    import numpy as np

    L = 5
    Z = [[0, 2, 0.1, 1],
         [4, 3, 0.3, 1],
         [1, 5, 0.4, 1],
         [6, 7, 0.8, 1]]

    Z = np.array(Z)
    Z = Z[:, :-1]

    # print(Z)

    class Node:
        def __init__(self, id, lc, rc):
            self.id = id
            self.lc = lc
            self.rc = rc

            self.is_leaf = self._is_node_leaf()

        def _is_node_leaf(self):
            return self.lc == 0 and self.rc == 0

        def __repr__(self):
            if self.is_leaf:
                return "[%s]" % (self.id)
            else:
                return "[%s -> (%s, %s)]" % (self.id, self.lc, self.rc)

    def find_max_node(roots):
        return max(roots, key=lambda x: x.id)

    def replace_children(node, roots):
        if isinstance(node, float) and node not in [r.id for r in roots]:
            return Node(node, 0, 0)

        if isinstance(node, float) and node in [r.id for r in roots]:
            id = node
            r = [r for r in roots if r.id == id][0]
            lc = r.lc
            rc = r.rc
            return Node(id, replace_children(lc, roots), replace_children(rc, roots))

        if isinstance(node, Node):
            node.lc = replace_children(node.lc, roots)
            node.rc = replace_children(node.rc, roots)

        return node

    def build_tree(Z, L):
        roots = []
        for i, z_i in enumerate(Z):
            print(i, z_i)
            lc = z_i[0]
            rc = z_i[1]
            roots.append((Node(L + i, lc, rc)))

        print(roots)

        root = find_max_node(roots)
        print("root", root)
        tree = replace_children(root, roots)
        print("tree", tree)

    build_tree(Z, L)


if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # main4()
    main6()
    # main7()
