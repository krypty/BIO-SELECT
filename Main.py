from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from datasets.MILE.MileDataset import MileDataset
from datasets.Golub99.GolubDataset import GolubDataset
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.feature_selection as fs


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


if __name__ == '__main__':
    main()
    # main2()
