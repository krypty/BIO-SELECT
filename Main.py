import numpy


def main():
    print("hello world")
    a = numpy.random.random()
    print(a)

    from numpy.random import choice

    list_of_candidates = [3, 8, 11]
    number_of_items_to_pick = 2
    probability_distribution = [0, 1, 0]
    for i in range(15):
        draw = choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution)
        print(draw)


if __name__ == '__main__':
    main()
