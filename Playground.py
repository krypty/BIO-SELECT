from time import sleep

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


if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    main4()
