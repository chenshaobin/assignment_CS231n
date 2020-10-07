from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    for i in range(num_train):
        current_row_scores = scores[i, :]
        # 通过减去当前分数中的最高值来保持数值稳定性
        shift_scores = current_row_scores - np.max(current_row_scores)
        # 计算当前数据的误差
        loss_current = -shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))
        loss += loss_current
        for j in range(num_classes):
            softmax_scores = np.exp(shift_scores[y[j]]) / np.sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_scores) * X[i, :]
            else:
                dW[:, j] += softmax_scores * X[i, :]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = np.dot(X, W)
    # print('scores shape:', scores.shape)  #(n,10)
    # print('maxScore shape:', np.max(scores, axis=1)[:, np.newaxis].shape) #(n,1)
    shift_scores = scores - np.max(scores, axis=1)[:, np.newaxis]
    # print('softmax_scores shape:', softmax_scores.shape)  #(n,10)
    # print('np.sum(np.exp(shift_scores), axis=1)[:, np.newaxis]:', np.sum(np.exp(shift_scores), axis=1)[:, np.newaxis].shape) #(n,1)
    softmax_scores = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1)[:, np.newaxis]


    dScore = softmax_scores
    dScore[range(num_train), y] = dScore[range(num_train), y] - 1

    dW = np.dot(X.T, dScore)
    dW /= num_train
    dW += 2 * reg * W
    # 找到正确类的分数值
    correct_class_scores = np.choose(y, shift_scores.T)
    loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
    loss = np.sum(loss) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
