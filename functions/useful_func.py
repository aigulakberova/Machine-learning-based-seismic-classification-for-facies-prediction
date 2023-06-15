
import matplotlib.pyplot as plt


def addlabels(y):
    
    for i in range(len(y)):
        plt.text(i, y[i], y[i], ha = 'center')