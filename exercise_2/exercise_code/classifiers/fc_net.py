import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros((1, num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        h1 = np.maximum(0,np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])), W1) + b1)  #
        scores = np.dot(h1, W2) + b2  #
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss

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
        dW1 = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dh1)  # (D,H)
        db1 = np.sum(dh1, axis=0, keepdims=True)  # (1,H)

        # Add the regularization gradient contribution
        dW2 += reg * W2
        dW1 += reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.params['W%d' % (i + 1)] = np.random.normal(scale=weight_scale, size=(dims[i], dims[i + 1]))
            self.params['b%d' % (i + 1)] = np.zeros(dims[i + 1])

            if self.use_batchnorm and i < len(dims) - 2:
                self.params['gamma%d' % (i + 1)] = np.ones(dims[i + 1])
                self.params['beta%d' % (i + 1)] = np.zeros(dims[i + 1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        input_ = X
        caches = []
        for i in range(self.num_layers - 1):
            n = i + 1
            input_, cache = FullyConnectedNet._layer_forward(input_, self.params['W%d' % n], self.params['b%d' % n],
                                                             self.params['gamma%d' % n] if 'gamma%d' % n
                                                                                           in self.params else None,
                                                             self.params['beta%d' % n] if 'beta%d' % n
                                                                                          in self.params else None,
                                                             use_batchnorm=self.use_batchnorm,
                                                             bn_params=self.bn_params[i] if self.bn_params else None,
                                                             use_dropout=self.use_dropout,
                                                             dropout_param=self.dropout_param)
            caches.append(cache)

        scores, cache_last_affine = affine_forward(input_, self.params['W%d' % self.num_layers],
                                                   self.params['b%d' % self.num_layers])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, d_scores = softmax_loss(scores, y)
        d_last_out, grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = affine_backward(d_scores,
                                                                                                     cache_last_affine)

        for i in range(self.num_layers - 1, 0, -1):
            d_last_out, grads['W%d' % i], grads['b%d' % i], d_gamma, d_beta = FullyConnectedNet._layer_backward(
                d_last_out, caches.pop(), use_batchnorm=self.use_batchnorm, use_dropout=self.use_dropout)
            if self.use_batchnorm:
                grads['gamma%d' % i], grads['beta%d' % i] = d_gamma, d_beta

        # noinspection PyTypeChecker
        loss += 0.5 * self.reg * sum(np.sum(self.params[param] ** 2) for param in self.params.keys()
                                     if param.startswith('W'))
        for param in self.params.keys():
            if param.startswith('W'):
                grads[param] += self.reg * self.params[param]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def _layer_forward(x, W, b, gamma=None, beta=None, use_batchnorm=False, bn_params=None, use_dropout=False,
                       dropout_param=None):
        out, cache_affine = affine_forward(x, W, b)
        if use_batchnorm:
            out, cache_bn = batchnorm_forward(out, gamma, beta, bn_params)
        else:
            cache_bn = None
        out, cache_relu = relu_forward(out)
        if use_dropout:
            out, cache_dropout = dropout_forward(out, dropout_param)
        else:
            cache_dropout = None
        return out, (cache_affine, cache_bn, cache_relu, cache_dropout)

    @staticmethod
    def _layer_backward(dout, cache, use_batchnorm=False, use_dropout=False):
        dx = dout
        cache_affine, cache_bn, cache_relu, cache_dropout = cache
        if use_dropout:
            dx = dropout_backward(dx, cache_dropout)
        dx = relu_backward(dx, cache_relu)
        if use_batchnorm:
            dx, d_gamma, d_beta = batchnorm_backward(dx, cache_bn)
        else:
            d_gamma = d_beta = None
        dx, dW, db = affine_backward(dx, cache_affine)


        return dx, dW, db, d_gamma, d_beta