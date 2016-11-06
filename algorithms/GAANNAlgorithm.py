import random

from sklearn.neural_network import MLPClassifier

from algorithms.Algorithm import Algorithm, NotSupportedException
from datasets.DatasetEncoder import DatasetEncoder
from datasets.DatasetSplitter import DatasetSplitter
from datasets.Golub99.GolubDataset import GolubDataset


class GAANAAlgorithm(Algorithm):
    def __init__(self, dataset, n):
        super(GAANAAlgorithm, self).__init__(dataset, n, name="GA ANN")

        ga = GA(dataset, pop_count=100, n_features_to_keep=n, n_generations=30)
        self._best_features = ga.run()

    def _get_best_features_by_score_unnormed(self):
        raise NotSupportedException()

    def get_best_features_by_rank(self):
        raise NotSupportedException()

    def get_best_features(self):
        return self._best_features


class GA:
    """
    inspired from : http://lethain.com/genetic-algorithms-cool-name-damn-simple/
    """

    @staticmethod
    def _assert_unique_features(individual):
        assert len(individual) == len(set(individual)), "Child contains non unique features %s" % individual

    def __init__(self, dataset, pop_count, n_features_to_keep, n_generations):
        """
        :type dataset: DatasetSplitter
        :type pop_count: int
        :type n_features_to_keep: int
        :type n_generations: int
        """
        self._pop_count = pop_count
        self._X_train = dataset.get_X_train()
        self._y_train = dataset.get_y_train()
        self._X_test = dataset.get_X_test()
        self._y_test = dataset.get_y_test()

        self.TOTAL_FEATURES = len(self._X_train[0])

        self._n_features_to_keep = n_features_to_keep
        self._n_generations = n_generations

    def run(self):
        pop = self._create_population()

        fitness_history = [self._grade_best_individual(pop), ]

        for _ in range(self._n_generations):
            pop = self._evolve(pop)
            fitness_history.append(self._grade_best_individual(pop))

        best_list_of_features = max(fitness_history, key=lambda x: x[1])
        return best_list_of_features[0]

    def _create_individual(self, n_features):
        """
        Create a member of the population. In this case a individual is a list of features

        Conditions to generate a valid individual
            1. each feature must be unique inside the individual
            2. the position of the features is not important
        """

        # we use random.sample to satisfy condition 1
        return random.sample(xrange(0, self.TOTAL_FEATURES), n_features)

    def _create_population(self):
        """
        Create a number of individuals (i.e. a population).

        count: the number of individuals in the population
        n_features: the number of features per individual
        """
        return [self._create_individual(self._n_features_to_keep) for _ in xrange(self._pop_count)]

    def _fitness_ann(self, individual):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3))
        clf.fit(self._X_train[:, individual], self._y_train)

        return clf.score(self._X_test[:, individual], self._y_test)

    def _grade_best_individual(self, pop):
        best_individual = (pop[0], self._fitness_ann(pop[0]))
        for x in pop:
            grd = self._fitness_ann(x)

            # maximize the score
            if grd > best_individual[1]:
                best_individual = (x, grd)
        return best_individual

    def _evolve(self, pop, retain=0.2, random_select=0.05, mutation_rate=0.01):
        graded = [(self._fitness_ann(x), x) for x in pop]
        graded = [x[1] for x in sorted(graded)]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]
        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random.random():
                parents.append(individual)

        # crossover parents to create children
        self._crossover(parents, pop)

        # mutate some individuals
        self._mutate(mutation_rate, parents)

        return parents

    # def crossover(parents, pop):
    #     parents_length = len(parents)
    #     desired_length = len(pop) - parents_length
    #     children = []
    #     while len(children) < desired_length:
    #         male = random.randint(0, parents_length - 1)
    #         female = random.randint(0, parents_length - 1)
    #         if male != female:
    #             male = parents[male]
    #             female = parents[female]
    #             half = len(male) / 2
    #             child = male[:half] + female[half:]
    #             children.append(child)
    #     parents.extend(children)

    def _crossover(self, parents, pop):
        def create_child(male, female):
            child_desired_length = len(male)

            # use the common features between the male and female
            inter_male_female = set().intersection(male, female)
            union_minus_inter_male_female = list(set().union(male, female).difference(inter_male_female))
            child = list(inter_male_female)

            # fill the missing features using both male and female features
            while len(child) < child_desired_length:
                feature = random.choice(union_minus_inter_male_female)

                if feature not in child:
                    child.append(feature)

            self._assert_unique_features(child)
            return child

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]

                child = create_child(male, female)
                children.append(child)

        parents.extend(children)

    # def mutate(mutation_rate, parents):
    #     for individual in parents:
    #         if mutation_rate > random.random():
    #             pos_to_mutate = random.randint(0, len(individual) - 1)
    #             # this mutation is not ideal, because it
    #             # restricts the range of possible values,
    #             # but the function is unaware of the min/max
    #             # values used to create the individuals,
    #             individual[pos_to_mutate] = random.randint(
    #                 min(individual), max(individual))

    def _mutate(self, mutation_rate, parents):
        individual_length = len(parents[0])

        for individual in parents:
            if mutation_rate > random.random():
                pos_to_mutate = random.randint(0, individual_length - 1)
                # this mutation is not ideal, because it
                # restricts the range of possible values,
                # but the function is unaware of the min/max
                # values used to create the individuals,

                new_feature = random.randint(0, self.TOTAL_FEATURES - 1)
                while new_feature in individual:
                    new_feature = random.randint(0, self.TOTAL_FEATURES - 1)

                individual[pos_to_mutate] = new_feature
                self._assert_unique_features(individual)


if __name__ == '__main__':
    import os

    os.chdir("..")

    ds = GolubDataset()

    # encode Dataset string classes into numbers
    ds_encoder = DatasetEncoder(ds)
    ds = ds_encoder.encode()
    ds = DatasetSplitter(ds, test_size=0.4)

    gaanaa = GAANAAlgorithm(ds, n=100)
    best_f = gaanaa.get_best_features()
    print("Best list of features by GAANAA %s" % best_f.__repr__())
