import lasagne
import theano.tensor as T
l_in = lasagne.layers.InputLayer((100, 50))
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200)
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10,
    nonlinearity=T.nnet.softmax)
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200,
       name="hidden_layer")
l = lasagne.layers.DenseLayer(l_in, num_units=100,
W=lasagne.init.Normal(0.01))

import theano
import numpy as np
W = theano.shared(np.random.normal(0, 0.01, (50, 100)))
l = lasagne.layers.DenseLayer(l_in, num_units=100, W=W)

W_init = np.random.normal(0, 0.01, (50, 100))
l = lasagne.layers.DenseLayer(l_in, num_units=100, W=W_init)
l = lasagne.layers.DenseLayer(l_in, num_units=100,
W=lasagne.init.Normal(0.01))
l1 = lasagne.layers.DenseLayer(l_in, num_units=100)
l2 = lasagne.layers.DenseLayer(l_in, num_units=100, W=l1.W)
x = T.matrix('x')
y = lasagne.layers.get_output(l_out, x)

print(lasagne.layers.get_output_shape(l_out))
f = theano.function([x], y)