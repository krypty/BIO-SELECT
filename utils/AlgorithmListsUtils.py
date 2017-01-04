import numpy as np
from collections import Counter
from itertools import groupby


def compute_most_popular_features(merged_list_of_lists):
    """
    Return a list of tuple (a,b) where a is the feature and b the number of times the feature a appears.
    This list is sorted by decreasing number of occurrence
    :param merged_list_of_lists:
    :return:
    """
    c = Counter()
    c.update(merged_list_of_lists)
    return c.most_common()


def compute_score_of_lists(merged_list_of_tuple, higher_is_better):
    """
    Compute the score of the outputs lists of an Algorithm.
    The type of the inner lists is: tuple(a, b) where a is the feature and
    b is the rank (int) (please use higher_is_better=False),
    or where b is the score (float) (please use higher_is_better=True)

    Warning: the returned list might be longer than the input lists

    Criteria of a good score:
        1. We want to favor the features that are selected multiple times by the same algorithm
        2. If two features have been selected the same amount of times, we want to be able to decide between them
        3. We prefer to have features that have been selected multiple times (even if they have a higher rank or a lower score) that keeping features that only appear once but with a better score/rang

    Answers to the criteria:
        1. We count the number of times the feature has been selected
        2. We take the mean of the rank/score of this feature among the lists
        3. We increase the weight of the number of occurrence compared to the score

    :param merged_list_of_tuple: list of list of features selected by an Algorithm
    :param higher_is_better: False if a higher score is better. True if a lower score is better
    :return: A "merged" list of tuple (a,b) where a is the feature and b is the computed score. This list is sorted
    by decreasing score, so [0] is the best feature/the feature with the highest score.
    """
    # boost the score if the feature is selected multiples times by the Algorithm
    occurrence_bonus = 1.2

    def keyfunc(x): return x[0]

    list_of_lists_sorted = sorted(merged_list_of_tuple, key=keyfunc)
    grouped_list = [list(j) for i, j in groupby(list_of_lists_sorted, key=keyfunc)]

    def score_func(n_occurrences, values):
        item_median = np.median(values)
        return (n_occurrences * occurrence_bonus) + item_median

    def score_func_lower_is_better(n_occurrences, values):
        item_median = np.median(values)
        # use 1/x function to inverse the score. Use 1 + median to avoid zero division
        return (n_occurrences * occurrence_bonus) + (1.0 / (1.0 + item_median))

    score_list = []
    for item in grouped_list:
        item_indices, item_values = zip(*item)

        item_index = item_indices[0]  # since this list contains n times the same index
        item_occurrences = len(item_indices)

        # use a different score function following the nature of the lists
        score_f = score_func if higher_is_better else score_func_lower_is_better

        item_score = score_f(item_occurrences, item_values)
        item_score_tuple = (item_index, item_score)

        score_list.append(item_score_tuple)

    # return the list sorted by
    return sorted(score_list, key=lambda x: x[1], reverse=True)
