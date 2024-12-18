import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate the model using standard metrics.
    
    Parameters:
    y_true (array-like): Ground truth target values.
    y_pred (array-like): Estimated targets as returned by a classifier.
    y_pred_proba (array-like, optional): Predicted probabilities. Defaults to None.
    
    Returns:
    dict: Dictionary containing various evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return metrics

def tf_evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate the model using TensorFlow operations.
    
    Parameters:
    y_true (tf.Tensor): Ground truth target values.
    y_pred (tf.Tensor): Estimated targets as returned by a classifier.
    y_pred_proba (tf.Tensor, optional): Predicted probabilities. Defaults to None.
    
    Returns:
    dict: Dictionary containing various evaluation metrics.
    """
    metrics = {
        'accuracy': tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32)),
        'precision': tf.reduce_sum(tf.math.multiply(y_true, y_pred)) / (tf.reduce_sum(y_pred) + 1e-7),
        'recall': tf.reduce_sum(tf.math.multiply(y_true, y_pred)) / (tf.reduce_sum(y_true) + 1e-7),
        'f1_score': 2 * tf.reduce_sum(tf.math.multiply(y_true, y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7),
    }
    
    if y_pred_proba is not None:
        # Note: TensorFlow doesn't have a built-in ROC-AUC metric.
        # For accurate AUC calculation, consider using scikit-learn's implementation.
        pass
    
    return metrics

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    
    Parameters:
    cm (numpy array): Confusion matrix.
    classes (list): List of class names.
    title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot the Receiver Operating Characteristic curve.
    
    Parameters:
    y_true (array-like): Ground truth target values.
    y_pred_proba (array-like): Predicted probabilities.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
