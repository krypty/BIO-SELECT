from matplotlib import pyplot as plt
import numpy as np
import itertools


class ConfusionMatrix:
    def __init__(self):
        pass

    @staticmethod
    def plot(cm, classes, title, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        """

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
