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

    for i in xrange(num_train):

      # initial score calculation scores= Wx
      scores = X[i].dot(W)
      # shift values for 'scores' to avoid numerical overflow which might occur 
      #due to blowing up of the expoential value by subracting the highest score
      #from all scores
      scores -= scores.max()
      scores_expo_sum = np.sum(np.exp(scores)) #denominator of every score value
      numerator = np.exp(scores[y[i]]) #numerator 
      #final loss for that particular data point
      loss += - np.log( numerator / scores_expo_sum)
      #gradient for softmax layer using cross entropy loss
      # reference : https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      # for correct class
      dW[:, y[i]] += (-1) * (scores_expo_sum - numerator) / scores_expo_sum * X[i]
      for j in xrange(num_classes):
          # pass correct class gradient
          if j == y[i]:
              continue
          # for incorrect classes
          dW[:, j] += np.exp(scores[j]) / scores_expo_sum * X[i]

    # averaging the loss over all the training examples
    loss /= num_train
    # adding the regularization loss to the total loss
    loss += reg * np.sum(W * W)

    #similarly averaging the gradient over all the training examples
    dW /= num_train
    #adding the regularization gradient to the total gradient term
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #calculate scores : initial score calculation scores= Wx
    scores = X.dot(W)
    # shift values for 'scores' to avoid numerical overflow which might occur 
    # due to blowing up of the expoential value by subracting the highest score
    # from all scores
    scores -= scores.max()
    scores = np.exp(scores)
    # calculating denominator for every data point by summing (along columns ->)
    scores_expo_sums = np.sum(scores, axis=1)
    numerators = scores[range(num_train), y] #numerators
    # loss calcuation by using reference : https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    loss = numerators / scores_expo_sums 
    loss = -np.sum(np.log(loss))/num_train + reg * np.sum(W * W)

    # gradient calculation
    # dividing the scores by their respective sums calculated before
    s = np.divide(scores, scores_expo_sums.reshape(num_train, 1))
    s[range(num_train), y] = - (scores_expo_sums - numerators) / scores_expo_sums
    dW = X.T.dot(s)

    #averaging the gradient over all the training examples
    dW /= num_train
    #adding the regularization gradient to the total gradient term
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
