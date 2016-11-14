import collections
import random

import functools
from sklearn.neural_network import MLPClassifier

from algorithms.Algorithm import Algorithm, NotSupportedException
from datasets.DatasetEncoder import DatasetEncoder
from datasets.DatasetSplitter import DatasetSplitter
from datasets.Golub99.GolubDataset import GolubDataset

import numpy as np

import multiprocessing as mp

from datasets.MILE.MileDataset import MileDataset


def fitness(instance, individual):
    """
    External wrapper method mandatory when using multiprocessing
    See http://stackoverflow.com/a/8805244
    :param instance: GA
    :param individual: list
    """
    return instance._fitness_ann(individual)


def timeit(method):
    import time

    def timed(*args, **kw):
        obj = args[0]
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        t_elapsed = te - ts
        obj._bench_times[method.__name__].append(t_elapsed)

        if obj._VERBOSE:
            print('%r %2.2f sec' % (method.__name__, t_elapsed))
        return result

    return timed


class GAANNAlgorithm(Algorithm):
    def __init__(self, dataset, n, verbose=False):
        super(GAANNAlgorithm, self).__init__(dataset, n, name="GA ANN")

        self.VERBOSE = verbose
        self._ga = GA(dataset, pop_count=100, n_features_to_keep=n, n_generations=30, verbose=verbose)
        self._best_features, self._score = self._ga.run()

    def _get_best_features_by_score_unnormed(self):
        raise NotSupportedException()

    def get_best_features_by_rank(self):
        raise NotSupportedException()

    def get_best_features(self):
        return self._best_features

    def get_score(self):
        return self._score


class GA:
    """
    inspired from : http://lethain.com/genetic-algorithms-cool-name-damn-simple/
    """

    @staticmethod
    def _assert_unique_features(features):
        assert len(features) == len(set(features)), "Child contains non unique features %s" % features

    def __init__(self, dataset, pop_count, n_features_to_keep, n_generations, verbose=False):
        """
        :type dataset: DatasetSplitter
        :type pop_count: int
        :type n_features_to_keep: int
        :type n_generations: int
        """
        self.POP_LENGTH = pop_count
        self._X_train = dataset.get_X_train()
        self._y_train = dataset.get_y_train()
        self._X_test = dataset.get_X_test()
        self._y_test = dataset.get_y_test()

        self.TOTAL_FEATURES = len(self._X_train[0])

        self._n_features_to_keep = n_features_to_keep
        self._n_generations = n_generations

        self._VERBOSE = verbose

        self._bench_times = collections.defaultdict(list)
        self._fitness_history = None

    @timeit
    def run(self):
        pop = self._create_population()

        self._fitness_history = []

        for _ in range(self._n_generations):
            pop = self._evolve(pop)

            pop = self._grade_pop(pop)

            best_individual = max(pop, key=lambda x: x[1])
            self._fitness_history.append(best_individual)

        best_individual_ever = max(self._fitness_history, key=lambda x: x[1])
        return best_individual_ever

    def _create_individual(self, n_features):
        """
        Create a member of the population. In this case a individual is a list of features

        Conditions to generate a valid individual
            1. each feature must be unique inside the individual
            2. the position of the features is not important
        """

        # we use random.sample to satisfy condition 1
        features_list = random.sample(xrange(0, self.TOTAL_FEATURES), n_features)
        score = 0  # 0 as it is not graded yet.
        return features_list, score

    @timeit
    def _create_population(self):
        """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the population
        n_features: the number of features per individual
        """
        return [self._create_individual(self._n_features_to_keep) for _ in xrange(self.POP_LENGTH)]

    def _fitness_ann(self, individual):
        features = individual[0]

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3))
        clf.fit(self._X_train[:, features], self._y_train)

        score = clf.score(self._X_test[:, features], self._y_test)
        return features, score

    @timeit
    def _grade_pop(self, pop):
        pool = mp.Pool(processes=mp.cpu_count())

        try:
            graded_pop = pool.map(functools.partial(fitness, self), pop)
        finally:
            pool.close()

        return graded_pop

    @timeit
    def _evolve(self, pop, n_elite=1, mutation_rate=0.01):
        elite_individuals = sorted(pop, key=lambda x: x[1], reverse=True)[:n_elite]
        # print("elites %.3f %s" % (elite_individuals[0][1], elite_individuals[0][0][:10]))

        # use these parents to create children, together they form a new population
        pop = self._crossover(pop)

        # to increase diversity, some individuals are mutated
        self._mutate(pop, mutation_rate)

        # keep elite individuals
        pop[:n_elite] = elite_individuals

        return pop

    def _select_parents(self, pop):
        """
        Selection method used is Fitness proportionate selection
        see : https://en.wikipedia.org/wiki/Fitness_proportionate_selection
        :param pop:
        :return: selected parents (1 male and 1 female)
        """

        probability_distribution = None
        sum_parents_score = sum([individual[1] for individual in pop])

        if sum_parents_score > 0:
            probability_distribution = map(lambda individual: individual[1] / float(sum_parents_score), pop)

        # chose n parents in the pop using a weight (probability_distribution).
        # With replace argument set to True, it allows choosing a parent multiple times

        mask_pop = range(self.POP_LENGTH)
        pop_indices = np.random.choice(mask_pop, 2, p=probability_distribution, replace=True)
        parents = [pop[i] for i in pop_indices]
        return parents

    @timeit
    def _crossover(self, pop):
        crossedover_pop = []

        for _ in xrange(self.POP_LENGTH):
            parents = self._select_parents(pop)
            male = parents[0]
            female = parents[1]
            child = self._create_child(male, female)
            crossedover_pop.append(child)

        return crossedover_pop

    def _create_child(self, male, female):
        # use only the feature lists
        male_feat = male[0]
        female_feat = female[0]

        child_desired_length = len(male_feat)

        # use the common features between the male and female
        inter_male_female = set(male_feat).intersection(female_feat)
        union_minus_inter_male_female = list(set(male_feat).union(female_feat).difference(inter_male_female))
        child = list(inter_male_female)

        # fill the missing features using both male and female features
        n_features_to_fill = child_desired_length - len(child)
        child.extend(random.sample(union_minus_inter_male_female, n_features_to_fill))

        self._assert_unique_features(child)
        score = 0  # 0 as it is not graded yet.
        return child, score

    @timeit
    def _mutate(self, pop, mutation_rate):
        for individual in pop:
            for gene in xrange(len(individual)):
                if mutation_rate > random.random():
                    new_feature = random.randint(0, self.TOTAL_FEATURES - 1)

                    while new_feature in individual[0]:
                        new_feature = random.randint(0, self.TOTAL_FEATURES - 1)

                    individual[0][gene] = new_feature
                    self._assert_unique_features(individual[0])


if __name__ == '__main__':
    import os

    os.chdir("..")

    # ds = GolubDataset()
    print("Loading dataset...")
    ds = MileDataset()
    print("Dataset loaded !")

    # encode Dataset string classes into numbers
    print("Encoding ds...")
    ds_encoder = DatasetEncoder(ds)
    ds = ds_encoder.encode()
    print("ds encoded !")
    ds = DatasetSplitter(ds, test_size=0.4)
    print("ds splitted !")

    print("Starting algorithm....")
    gaanaa = GAANNAlgorithm(ds, n=100, verbose=True)
    best_f = gaanaa.get_best_features()
    print("Best list of features by GAANAA %s" % best_f.__repr__())

    # stability test
    features = [6530, 5078, 784, 5396, 6934, 1183, 1959, 4523, 4782, 3251, 4425, 5562, 6460, 1981, 318, 6591, 1091,
                7113, 4690, 1878, 6487, 6880, 2704, 1110, 5230, 2420, 5439, 6510, 4243, 5779, 4041, 3857, 5165, 2682,
                680, 5459, 3984, 1887, 6613, 2965, 241, 3393, 6674, 3375, 3426, 6704, 990, 1439, 5398, 4024, 4260, 6509,
                4317, 7057, 1357, 2901, 6169, 6156, 6877, 1968, 6627, 5214, 6285, 3564, 2020, 1593, 2023, 3723, 4532,
                3367, 5502, 5599, 767, 2426, 1687, 1615, 4624, 5753, 4490, 3637, 4126, 4688, 3424, 420, 432, 630, 2619,
                926, 4586, 3147, 5947, 2856, 266, 1844, 1881, 2142, 1805, 560, 4094, 2401]
    assert len(features) == len(set(features))

    scores = []
    for _ in range(20):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3))
        clf.fit(ds.get_X_train()[:, features], ds.get_y_train())

        score = clf.score(ds.get_X_test()[:, features], ds.get_y_test())
        print(score)

        scores.append(score)

    print("std", np.std(scores))
    print("mean", np.mean(scores))
