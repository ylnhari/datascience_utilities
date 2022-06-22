"""Module to Calculate Classification and regression metrics.
"""

from sklearn import metrics

def get_classifier_metrics(y_test, y_prob, threshold=0.5):
    """
        Function to calculate all pre chosen metrics required to evaluate a classifier
        parameter
        ---------
        y_test(numpyarray/pandas series object) : test labels
        y_prob(numpy array) : Probability Predictions made using model
        threshold(float): threshold at which points should be classified into either classes.
        returns
        -------
        all calculated metrics are returned as a python dict
    """
    y_pred = get_labels(y_prob, threshold)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    true_negatives = confusion_matrix[0][0]
    true_positives = confusion_matrix[1][1]
    false_negatives = confusion_matrix[1][0]
    false_positives = confusion_matrix[0][1]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision_positive = metrics.precision_score(y_test, y_pred, pos_label=1)
    precision_negative = metrics.precision_score(y_test, y_pred, pos_label=0)
    recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
    recall_specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
    f1_positive = metrics.f1_score(y_test, y_pred, pos_label=1)
    f1_negative = metrics.f1_score(y_test, y_pred, pos_label=0)
    fpr, tpr, thresholds1 = metrics.roc_curve(y_test, y_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    metric_dict = \
        {'true_negatives': true_negatives,
         'true_positives': true_positives,
         'false_negatives': false_negatives,
         'false_positives': false_positives,
         'accuracy': accuracy,
         'precision_positive': precision_positive,
         'precision_negative': precision_negative,
         'recall_sensitivity': recall_sensitivity,
         'recall_specificity': recall_specificity,
         'f1_positive': f1_positive,
         'f1_negative': f1_negative,
         'auc': auc}
    return metric_dict


def get_regression_metrics(y_test, y_pred):
    """
    Function to calculate all pre chosen regression performance metrics
    ------------
    Parameters:
    ------------
        y_test(numpy array/pandas series object) : Actual Test Data
        y_pred(numpy array/pandas series object): predictions obtained on test data
        return(Python dict)
        returns metrics dictionary
    """
    metric_dict = {'Mean Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
                   'Mean Squared Error': metrics.mean_squared_error(y_test, y_pred),
                   'Root Mean Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                   'r2': metrics.r2_score(y_test, y_pred)}

    return metric_dict
