import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


class Dendrogram:
    def __init__(self, lists, lists_labels, metric):
        self._lists = lists
        self._lists_labels = lists_labels
        self._metric = metric

        self._lists_mask = self._compute_masks()
        self._Z = linkage(self._lists_mask, metric=metric)

    def show(self):
        plt.figure(figsize=(12, 4))
        plt.title('Hierarchical Clustering Dendrogram\n' + self._metric)
        plt.xlabel('lists of selected features')
        plt.ylabel('distance')

        self._fancy_dendrogram(
            self._Z,
            labels=self._lists_labels,
            leaf_rotation=90.,  # rotates the x axis labels,
        )
        plt.show()

    def _compute_masks(self):
        # Use the union of all the selected features in all the lists to create the mask.
        # Using the union allows us to reduce the size of the mask by ignoring the features that are not in any lists
        # The mask is simply a list of booleans. True if the feature is in the list, False otherwise
        # Warning: we lose the sorting with the union
        lists_union = list(reduce(lambda a, b: set(a).union(set(b)), self._lists))

        l_mask = [self._get_mask_of_features(f, lists_union) for f in self._lists]
        l_mask = np.array(l_mask)

        return l_mask

    @staticmethod
    def _get_mask_of_features(features, union):
        return [u_i in features for u_i in union]

    @staticmethod
    def _fancy_dendrogram(*args, **kwargs):
        """
        Source: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        """
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
