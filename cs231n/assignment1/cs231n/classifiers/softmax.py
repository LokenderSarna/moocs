import numpy as np
from random import shuffle

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
  loss = np.float64(0.0)
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  product = X.dot(W)
  for i in range(num_train):
      scores =  X[i].dot(W)
      scores -= np.max(scores)
      scores = np.exp(scores) / np.sum(np.exp(scores))
      for c in range(num_classes):
          if c == y[i]:
              dW[:, c] += X[i,:] * (scores[c] - 1)
          else:
              dW[:,c] += scores[c] * X[i,:]
            
      # adjusting for numerical stability
      loss += -np.log(scores[y[i]])
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss / np.float64(num_train)
  loss += 0.5 * reg * np.sum(W * W)

  dW = dW / np.float64(num_train)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = np.float64(0.0)
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # N,C
  norm_scores = X.dot(W) - np.max(scores, axis=1).reshape((num_train, 1))
  exp_scores= np.exp(norm_scores)
  prob = exp_scores / np.sum(exp_scores, axis=1).reshape((num_train, 1))
  losses = - np.log(prob[np.arange(num_train), y])
  loss = np.mean(losses)
  loss +=  0.5 * reg * np.sum(W * W)

  # Gradient calculation
  prob[np.arange(num_train), y] -= 1
  dW = X.T.dot(prob)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

