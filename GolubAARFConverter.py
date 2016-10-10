import arff
import os
import pandas as pd


def main(input_file, output_folder):
    """
    Original data have been downloaded here ("Leukemia" section): http://eps.upo.es/bigs/datasets.html

    Convert Golub arff dataset into N (where N is the numbers of samples) csv files.
    CSV files are formatted with 2 columns and M lines (number of features):
        ID_REF  VALUE
        feat1   val1
        feat2   val2
        ...     ...
        featM   valM

    Requirements : Pandas, LIAC-ARFF

    :param input_file: original source file in *.arff format
    :param output_folder: output folder where the csv files will be saved
    :return: nothing
    """
    decoder = arff.ArffDecoder()
    f = open(input_file)
    data = decoder.decode(f, encode_nominal=True)

    labels = [d[0] for d in data["attributes"][:-1]]

    labels_lookup_table = ["ALL", "AML"]

    for sample in range(len(data["data"])):
        class_name = labels_lookup_table[data["data"][sample][-1]]
        sample_filename = "sample_%s_%s.csv" % (sample, class_name)
        sample_filename = output_folder + os.sep + sample_filename
        print(sample_filename)

        write_sample(filename=sample_filename, data=data["data"][sample][:-1], labels=labels)


def write_sample(filename, data, labels):
    df = pd.Series(data, index=labels)
    df.to_csv(filename, sep='\t', encoding='utf-8', header=True, index_label=["ID_REF", "VALUE"])


if __name__ == '__main__':
    """
    Example of usage
    The original data in *.arff have been saved in a 'data/golub' folder
    """

    # Generate train files
    input_file = r'./data/golub99/leukemia_train_38x7129.arff'
    output_folder = r'./data/golub99/processed/train'
    main(input_file=input_file, output_folder=output_folder)

    # Generate test files
    input_file = r'./data/golub99/leukemia_test_34x7129.arff'
    output_folder = r'./data/golub99/processed/test'
    main(input_file=input_file, output_folder=output_folder)
