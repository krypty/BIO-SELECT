from merge.techniques.UnionSubsetMerger import UnionSubsetMerger


class TestUnionSubsetMerger:
    def test_merge(self):
        a = [1, 2, 6, 7, 4]
        b = [1, 2, 5, 8, 9]

        expected = {1, 2, 4, 5, 6, 7, 8, 9}
        print(expected)

        susm = UnionSubsetMerger([a, b])
        merged = susm.merge()
        print(merged)

        assert merged == expected
