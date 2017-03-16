"""Loss functions."""

import tensorflow as tf
import numpy as np
#import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    diff = np.absolute(y_true - y_pred)
    loss = np.zeros_like(diff)
    pos = np.where(diff <= max_grad)
    loss[pos] =  0.5*np.power(diff[pos],2)
    pos = np.where(diff > max_grad)
    loss[pos] =  max_grad * (diff[pos] - 0.5*max_grad)
    return loss


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    loss = huber_loss(y_true, y_pred, max_grad)
    return np.mean(loss)
