from merge.simple.SimpleUnionSubsetMerger import SimpleUnionSubsetMerger


class TestSimpleUnionSubsetMerger:
    def test_merge(self):
        a = [1, 2, 6, 7, 4]
        b = [1, 2, 5, 8, 9]

        expected = {1, 2, 4, 5, 6, 7, 8, 9}
        print(expected)

        susm = SimpleUnionSubsetMerger([a, b])
        merged = susm.merge()
        print(merged)

        assert merged == expected
