from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # here we first reshape the input to the above mentioned form which is 
    #same as converting the images to each row where total rows is the size of
    #minibatch. Hence we reshape it using x.shape[0]
    X= x.reshape(x.shape[0],-1)
    #like any other linear classifier we calculate the ouput by y= wx+b form
    # we multiply X,w to adjust the dimensions of the matrices 
    out = np.matmul(X,w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #
    #I have used the following pdf as reference for calculating the gradients
    # http://cs231n.stanford.edu/handouts/linear-backprop.pdf from this resource
    # I have made use of the dw, dx calculations.
    #Just like the previous function, we first reshape our input x 
    X= x.reshape(x.shape[0],-1)
    # We start off by calculating the gradient wrto. input by multiplying the
    #weights with the upstream gradient and reshaping it into the same shape as
    #our input which is (10,2,3)
    dx= np.matmul(dout, w.T).reshape(x.shape)
    # We calculate the gradient wrto. weights which is matrix multiplication of 
    # reshaped input with upstream gradient 
    dw = np.matmul(X.T ,dout)
    # Gradient wrto bias is 1 multiplied by upstream derivative thus ends up
    # beoming summation of dout
    #print(dout) 
    db = np.sum(dout, axis=0)
    #print(db)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Implementation of relu is straight forward, which is max(x, 0)
    # Hence out outputs x if x>0 is true(=1) else, 0 (~False)
    out = x * (x > 0) 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Implementation of relu backward is also very straight forward. We multiply
    # the upstream gradient with 1 if x is positive, else it becomes 0. 
    # some reference taken from link: https://datascience.stackexchange.com/questions/19272/deep-neural-network-backpropogation-with-relu
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # most of the code in the following lines have been taken from the
    #svm_vectorised implementation from classifiers
    ##store the total number of training examples into num_train
    num_train = x.shape[0]
    #read correct scores into a column array of height N 
    correct_class_scores = x[list(range(num_train)), y]
    
    # here we complete the calculation of margins by subracting respective 
    # correct_scores and adding delta (since margin = Score_j - corectscore + delta)
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1)

    # make sure correct scores themselves don't contribute to loss function, therefore
    # we set them to zero
    margins[list(range(num_train)), y] = 0

    # construct final loss function by summing over the entires scores matrix containing
    #margins by using function max(0,margins) and averaging it
    loss = np.sum(margins) / num_train
    num_pos = np.sum(margins > 0, axis=1)

    # calculation of gradient wrto input
    dx = np.zeros_like(x)
    # we set the derivatives at all points to 1
    dx[margins > 0] = 1
  
    #then we subract from points which have correct class scores to make them zero
    # previously we used masking in classifiers, here we experiment differently and
    # finally average over all training examples
    dx[list(range(num_train)), y] -= num_pos
    dx /= num_train
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # this is almost exact replication of the softmax classifier
    # here we find the numerators of softmax 
    scores = np.exp(x - np.max(x, axis=1, keepdims=True)) #numerator
    # here we find the denominiator by summming all probabilities of that row
    scores /= np.sum(scores, axis=1, keepdims=True) # denominator
    # we save the total number of training examples into num_train
    num_train = x.shape[0]
    #loss calcuation by using reference : https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    loss = -np.sum(np.log(scores[np.arange(num_train), y])) / num_train
    
    # here we start off by recalculating scores,then dividing the scores by their respective sums calculated before
    dx=np.exp(x - np.max(x, axis=1, keepdims=True))
    dx /= np.sum(dx, axis=1, keepdims=True)
    dx[np.arange(num_train), y] -= 1
    #averaging the gradient over all the training examples
    dx /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
