import numpy as np
from sklearn.svm import LinearSVC, SVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    model=LinearSVC(random_state = 0, C=0.1)
    model.fit(train_x,train_y)
    pred_test_y=model.predict(test_x)
    return pred_test_y
    raise NotImplementedError


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    model=LinearSVC(multi_class='ovr',random_state = 0, C=0.1)
    model.fit(train_x,train_y)
    pred_test_y=model.predict(test_x)
    return pred_test_y
    raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)


def multi_class_svm_cube(train_x, train_y, test_x):
    """
    Use the sklearn package and build the SVM model on your local machine. Use random_state = 0, kernel = 'poly', degree = 3
    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    model=SVC(random_state = 0, kernel = 'poly', degree = 3)
    model.fit(train_x,train_y)
    pred_test_y=model.predict(test_x)
    return pred_test_y


def multi_class_svm_rbf(train_x, train_y, test_x):
    """
    Use random_state = 0, kernel = 'rbf', and default values for other parameters.
    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    model=SVC(random_state = 0, kernel = 'rbf', gamma='auto')
    model.fit(train_x,train_y)
    pred_test_y=model.predict(test_x)
    return pred_test_y
