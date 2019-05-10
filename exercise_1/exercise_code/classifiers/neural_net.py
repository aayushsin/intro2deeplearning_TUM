"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        h1 = np.maximum(0,np.dot(X, W1) + b1)  #
        scores = np.dot(h1, W2) + b2  #
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N
        # Backprop into W2 and b2
        dW2 = np.dot(h1.T, dscores)  # (H,C)
        db2 = np.sum(dscores, axis=0, keepdims=True)  # (1,C)
        # Backprop into hidden layer
        dh1 = np.dot(dscores, W2.T)  # (N,H)
        # Backprop into ReLU non-linearity
        dh1[h1 <= 0] = 0
        # Backprop into W1 and b1
        dW1 = np.dot(X.T, dh1)  # (D,H)
        db1 = np.sum(dh1, axis=0, keepdims=True)  # (1,H)

        # Add the regularization gradient contribution
        dW2 += reg * W2
        dW1 += reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)
        #mu=0.90
        #mu_increase=1.0
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]        # (batch_size,D)
            y_batch = y[sample_index]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            v_W2 = -learning_rate * grads['W2']
            self.params['W2'] += v_W2
            v_b2 =  -learning_rate * grads['b2']
            self.params['b2'] += v_b2.flatten()
            v_W1 = -learning_rate * grads['W1']
            self.params['W1'] += v_W1
            v_b1 = -learning_rate * grads['b1']
            self.params['b1'] += v_b1.flatten()
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay
                #mu *= mu_increase

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        h1 = np.maximum(0,np.dot(X, self.params['W1']) + self.params['b1'])
        scores = np.dot(h1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 
    results = {}
    best_val = -1
    learning_rates = [3e-3,1e-3,9e-4]
    regularization_strengths = [0.009,0.01,0.06]
    num_iters = 2000
    input_size = 32 * 32 * 3
    hidden_size = [120,80,200]
    learning_rate_decays = 0.95
    num_classes = 10
    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    test_val_acc = 0.3
    test_train_acc = 0.3
    for i in range(len(learning_rates)):
        for j in range(len(regularization_strengths)):
            for k in range(len(hidden_size)):
                nnet = TwoLayerNet(input_size, hidden_size[k], num_classes)
                stats = nnet.train(X_train, y_train, X_val, y_val, num_iters=num_iters, batch_size=200, learning_rate=learning_rates[i], learning_rate_decay=learning_rate_decays, reg=regularization_strengths[j], verbose=False)
                y_train_pred_sgd = nnet.predict(X_train)
                y_val_pred_sgd = nnet.predict(X_val)
                val_accuracy = np.mean(y_val == y_val_pred_sgd)
                train_accuracy = np.mean(y_train == y_train_pred_sgd)
                results[(learning_rates[i], regularization_strengths[j], hidden_size[k])] = (train_accuracy, val_accuracy)
                if (val_accuracy > test_val_acc):
                    best_net = nnet
                    test_val_acc = val_accuracy
                    test_train_acc = train_accuracy
                    best_val = val_accuracy
    # for (lr, reg, hid) in sorted(results):
    #     train_accuracy, val_accuracy = results[(lr, reg)]
    #     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
    #                     lr, reg, train_accuracy, val_accuracy))
    #     print('best validation accuracy achieved during validation: %f' % best_val)
    a=1
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net