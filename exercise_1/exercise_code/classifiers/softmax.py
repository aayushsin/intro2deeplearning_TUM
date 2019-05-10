"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    # ############################################################################

    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))

    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    #
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0  # 每行只有正确的类别处为1，其余为0
    #
    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train


    dW += reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    num_train, dim = X.shape
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
 
    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C

    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)


    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [30e-7,40e-7,50e-7]
    regularization_strengths = [1e1,5e2, 7e2]
    num_iters = 1000

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    test_val_acc=0.3
    test_train_acc=0.3
    for i in range(len(learning_rates)):
        for j in range(len(regularization_strengths)):
            softmax = SoftmaxClassifier()
            loss_history = softmax.train(X_train,y_train,learning_rate=learning_rates[i],reg=regularization_strengths[j], num_iters=1500, verbose=False)
            y_train_pred_sgd = softmax.predict(X_train)
            y_val_pred_sgd = softmax.predict(X_val)
            val_accuracy=np.mean(y_val == y_val_pred_sgd)
            train_accuracy=np.mean(y_train == y_train_pred_sgd)
            results[(learning_rates[i], regularization_strengths[j])] = ( train_accuracy,val_accuracy)
            if (val_accuracy>test_val_acc):
                best_softmax= softmax
                test_val_acc=val_accuracy
                test_train_acc=train_accuracy
                best_val=val_accuracy

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        all_classifiers.append((softmax, val_accuracy))
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
