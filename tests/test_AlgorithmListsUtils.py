from utils.AlgorithmListsUtils import *


class TestAlgorithmListsUtils:
    @staticmethod
    def _is_tuple_equals(output_tuple, expected_tuple):
        epsilon = 1e-4
        ok = True
        ok = ok and output_tuple[0] == expected_tuple[0]
        ok = ok and abs(output_tuple[1] - expected_tuple[1]) <= epsilon
        return ok

    def test_compute_most_popular_features(self):
        print("Theses lists of list of features: ")
        list_run1 = [1, 2, 3]
        list_run2 = [1, 2, 4]
        list_all_runs = []
        list_all_runs.extend(list_run1)
        list_all_runs.extend(list_run2)
        print(">> ", list_all_runs)

        print("Give: ")

        computed_list_of_lists = compute_most_popular_features(list_all_runs)
        print(computed_list_of_lists)

        assert computed_list_of_lists[0] == (1, 2)
        assert computed_list_of_lists[1] == (2, 2)
        assert computed_list_of_lists[2] == (3, 1)
        assert computed_list_of_lists[3] == (4, 1)

    def test_compute_list_of_ranks(self):
        print("Theses lists of ranks: ")

        def ranks_to_tuples(r): return [(v, k) for k, v in enumerate(r)]

        rank_run1 = ranks_to_tuples([1, 2, 6, 3])
        rank_run2 = ranks_to_tuples([1, 2, 3, 4])
        rank_run3 = ranks_to_tuples([2, 6, 8, 1])

        print(rank_run1)
        print(rank_run2)
        print(rank_run3)
        rank_all_runs = []
        rank_all_runs.extend(rank_run1)
        rank_all_runs.extend(rank_run2)
        rank_all_runs.extend(rank_run3)
        print(">> ", rank_all_runs)

        print("Give: ")
        computed_list_of_ranks = compute_score_of_lists(rank_all_runs, higher_is_better=False)
        print(computed_list_of_ranks)

        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[0], (1, 4.5))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[1], (2, 2.25))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[2], (6, 1.2))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[3], (3, 0.857142))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[4], (8, 0.5))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_ranks[5], (4, 0.375))

        # keep the list as long as the originals, assuming all lists are the same length for the given algorithm
        computed_list_of_ranks = computed_list_of_ranks[:len(rank_run1)]
        print(computed_list_of_ranks)

    def test_compute_list_of_scores(self):
        print("Theses lists of scores: ")
        score_run1 = [(1, 0.9), (2, 0.8), (6, 0.6), (3, 0.4)]
        score_run2 = [(1, 0.5), (2, 0.4), (3, 0.4), (4, 0.1)]
        score_run3 = [(2, 0.9), (8, 0.8), (6, 0.5), (1, 0.4)]

        print(score_run1)
        print(score_run2)
        print(score_run3)
        score_all_runs = []
        score_all_runs.extend(score_run1)
        score_all_runs.extend(score_run2)
        score_all_runs.extend(score_run3)
        print(">> ", score_all_runs)

        print("Give: ")
        computed_list_of_scores = compute_score_of_lists(score_all_runs, higher_is_better=True)
        print(computed_list_of_scores)

        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[0], (2, 3.6))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[1], (1, 2.25))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[2], (6, 1.65))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[3], (3, 1.2))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[4], (8, 1.2))
        assert TestAlgorithmListsUtils._is_tuple_equals(computed_list_of_scores[5], (4, 0.15))

        # keep the list as long as the originals, assuming all lists are the same length for the given algorithm
        computed_list_of_scores = computed_list_of_scores[:len(score_run1)]
        print(computed_list_of_scores)
