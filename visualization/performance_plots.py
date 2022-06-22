"""
This module helps to plot all performance plots of classification and regression models.
"""
__author__ = "Y.L.N.Hari"
__email__ = "ylnharimailme@gmail.com"


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np


def plot_roc_curve(y_test, y_prob):
    """
    Function to calculate all pre chosen metrics required to evaluate a classifier
    ------------
    Parameters:
    ------------
        y_test(numpyarray/pandas series object) : test labels
        y_prob(numpy array) : Probability Predictions made using model
    return: matplotlib.pypot object
        returns roc plot()
    """
    sns.set_style('darkgrid')
    # false positive rate, true positive rate, thresholds
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob[:, 1])
    # auc score
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    # plot auc
    plt.plot(fpr, tpr, color='blue', label='Test ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # plt.savefig('roc_curve.png')
    inf = plt.gcf()
    plt.show()
    return inf


def plot_precision_recall_curve(y_test, y_prob):
    """
        Function to calculate all pre chosen metrics required to evaluate a classifier
        ------------
       Parameters:
       ------------
            y_test(numpyarray/pandas series object) : test labels
            y_prob(numpy array) : Probability Predictions made using model
        return: matplotlib.pypot object
            returns precision-recall plot()
        """
    pre, rec, thr = metrics.precision_recall_curve(y_test, y_prob[:, 1])
    plt.figure(figsize=(8, 4))
    plt.plot(thr, pre[:-1], label='precision')
    plt.plot(thr, rec[1:], label='recall')
    plt.xlabel('Threshold')
    plt.title('Precision & Recall vs Threshold', c='r', size=16)
    plt.legend()
    # plt.savefig('pr_curve.png')
    inf = plt.gcf()
    plt.show()
    return inf


def plot_confusion_matrix(y_test, y_pred, labels_c=[0, 1]):
    """
       Helper Function To save Plot of confusion matrix
       ------------
       Parameters:
       ------------
       y_true    :Actual Labels (npy array)
       y_pred    :Predicted Labels (npy array)
       labels_c(list)  :Label Names (helps in case of multi class classification_
    """
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d', annot_kws={"color": 'black'}, cmap='Blues')
    # labels_c, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix - Image Classification')
    ax.xaxis.set_ticklabels(labels_c)
    ax.yaxis.set_ticklabels(labels_c)
    # plt.savefig('confusion_matrix.png')
    inf = plt.gcf()
    plt.show()
    return inf


def plot_threshold_vs_r2(recall, r2, threshold):
    sns.set_style('darkgrid')
    plt.figure(figsize=(8, 8))
    plt.plot(recall, r2, color='blue', label='r2 at different thresholds')
    for i in range(0, len(recall)):
        plt.annotate(threshold[i], (recall[i], r2[i]))
    plt.xlabel('recall', size=14)
    plt.ylabel('R2', size=14)
    plt.legend(loc='lower right')
    inf = plt.gcf()
    plt.show()
    return inf


def plot_dist_plot_for_two_columns(y, y_trans):
    """
        Helper function to plot distribution plot for two columns
        Parameters
        y(numpy array) : First Column
        y_trans(numpy array) : Second Column
        return:
        Nothing
    """
    import seaborn as sns
    f, (ax0, ax1) = plt.subplots(1, 2)
    sns.distplot(y, ax=ax0)
    sns.distplot(y_trans, ax=ax1)

    ax0.set_ylabel('Probability')
    ax0.set_xlabel('Target')
    ax0.set_title('Target distribution')

    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Target')
    ax1.set_title('Transformed target distribution')
    f.suptitle("Reject Rate data", y=0.06, x=0.53)


def plot_residuals(y_test, y_pred):
    """
    Helper Function to plot residuals.
    Parameters
        y_true    :Actual Data (npy array)
        y_pred    :Predicted Data(npy array)
    returns matplotlib.pyplot object
        returns residual plot
    """

    ax = plt.subplot()
    sns.residplot(y_test, y_pred)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix - Image Classification')
    ax.xaxis.set_ticklabels()
    ax.yaxis.set_ticklabels()
    plt.show()
    return plt


def plot_line_of_fit(y_test, y_pred):
    """
    Helper Function to plot residuals.
    Parameters
        y_true    :Actual Data (npy array)
        y_pred    :Predicted Data(npy array)
    returns matplotlib.pyplot object
        returns line of fit
    """
    x = np.linspace(0, y_test.size, 1)
    y_i = np.argsort(y_test)
    y_test = y_test[y_i]
    y_pred = y_pred[y_i]
    dy = y_pred - y_test
    fig, ax = plt.subplots()
    ax.plot(x, y_test)
    ax.scatter(x, y_test + dy)
    ax.vlines(x, y_test, y_test + dy)
    ax.set_xlabel('data points')
    ax.set_ylabel('Reject Rate')
    ax.set_title('Line of Fit')
    plt.show()
    return plt
