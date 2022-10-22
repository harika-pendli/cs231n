from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributors_count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #from the gradient of the SVM loss function we know that it has two parts 
                #one with respec to wj and the other with respect to wyi which represent
                #the partial derivatives for incorrect and correct class respectively
                # incorrect class gradient part
                dW[:, j] += X[i]
                # correct class gradient part
                dW[:, y[i]] += (-1) * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Adding derivative of regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #we make use of the intermediate values calculated of loss functions  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #store the total number of training examples into num_train, just
    #like the previous function
    num_train = X.shape[0]

    # scores: A numpy array of shape (N, C) containing scores given by linear
    #function scores= Wx
    scores = X.dot(W)

    # read correct scores into a column array of height N 
    correct_score = scores[list(range(num_train)), y]
    #print(correct_score.shape,"before correct_score shape") gives (500,)
    correct_score = correct_score.reshape(num_train, -1)
    #print(correct_score.shape,"After reshaping") gives (500,1)

    # subtract correct scores from score matrix and add margin
    # here we complete the calculation of margins by subracting respective 
    # correct_scores and adding delta (since margin = Score_j - corectscore + delta)
    scores += 1 - correct_score
    # make sure correct scores themselves don't contribute to loss function, therefore
    # we set them to zero
    scores[list(range(num_train)), y] = 0
    
    # construct final loss function by summing over the entires scores matrix containing
    #margins by using function max(0,margins) and averaging it
    loss = np.sum(np.fmax(scores, 0)) / num_train
    # to the final loss, add the regulariation loss 
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # we create a mask for scores to obtain values that contribute to loss 
    X_mask = np.zeros(scores.shape)
    # we set the derivatives at that point to 1
    X_mask[scores > 0] = 1
    # we set the derivatives at correct labels to count of loss contributors, which 
    # we got from above
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    # Final non-averaged  gradients 
    dW = X.T.dot(X_mask)
    # averaging over all training examples
    dW /= num_train
    # finally, adding the regularization gradient term 
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
