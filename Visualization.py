"""
Functions for plotting the results
"""
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn import metrics
import numpy as np

def residuals_plot(pred,test):
    """
    Plot residuals plot
    """
    residuals = test - pred
    plt.hist(residuals,bins=100)
    plt.xlim(-5, 5)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    print(root_mean_squared_error(pred,test))
    print(np.mean(residuals))

def confusion_matrix_plot(true_positions, predicted_position):
    """
    Plot confusion matrix
    """
    confusion_matrix = metrics.confusion_matrix(true_positions, predicted_position)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1,2,3,4,5,6])
    cm_display.plot()
    plt.show()