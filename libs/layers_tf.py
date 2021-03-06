from inits_tf import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = None #placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        #output=tf.nn.l2_normalize(output,axis=1)
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



class GraphConvolutionBatch(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False,nkernel=1,**kwargs):
        super(GraphConvolutionBatch, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']  
        
        self.bias = bias
        self.nkernel=nkernel  

        self.istrain = placeholders['istrain']                 

        with tf.variable_scope(self.name + '_vars'):
            for i in range(0,self.nkernel):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))
                  
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        supports = list()

        # convolve
        for i in range(0,self.nkernel):            
            s0=tf.matmul(self.support[:,i,:,:],x)
            output=tf.tensordot(s0,self.vars['weights_' + str(i)],[2, 0])
            supports.append(output)           

        output = tf.add_n(supports)
        
        # bias
        if self.bias:
            output += self.vars['bias']
                 

        out=self.act(output)
        self.out=out
        return out

class GraphConvolutionwithDephSepBatch(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=(0.,0.),act=tf.nn.relu, bias=False,firstDSWS=True,
                 isdepthwise=True,featureless=False, **kwargs):
        super(GraphConvolutionwithDephSepBatch, self).__init__(**kwargs)

        self.isdropout=dropout
        if dropout[0] or dropout[1]:
            self.dropout = placeholders['dropout']            
        else:
            self.dropout = 0.

        self.firstDSWS=firstDSWS
        self.isdepthwise=isdepthwise
        self.act = act
        self.support = placeholders['support']        
        self.featureless = featureless
        self.bias = bias                  

        with tf.variable_scope(self.name + '_vars'):
            for i in range(0,self.support.shape[1]):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        if self.isdropout[0]:
            x = tf.nn.dropout(x, 1-self.dropout)            
        # convolve
        supports = list()
        for i in range(0,self.support.shape[1]):
            if self.isdropout[1]:
                tmp=tf.nn.dropout(self.support[:,i,:,:], 1-self.dropout) 
                s0=tf.matmul(tmp,x) 
            else:
                s0=tf.matmul(self.support[:,i,:,:],x)

            s0=tf.tensordot(s0,self.vars['weights_' + str(i)],[2, 0])
            supports.append(s0)
        output = tf.add_n(supports)           

        # bias
        if self.bias:
            output += self.vars['bias']        

        return self.act(output)


class AggLayer(Layer):
    
    def __init__(self, placeholders,method='mean',**kwargs):
        super(AggLayer, self).__init__(**kwargs)
        self.ND=placeholders['nnodes'] 
        self.method=method
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.method=='mean':
            output=tf.reduce_sum(x,1)/self.ND   
        elif self.method=='max':
            output=tf.reduce_max(x,1)
        elif self.method=='sum':
            output=tf.reduce_sum(x,1)
        else:
            output=tf.concat([tf.reduce_sum(x,1)/self.ND, tf.reduce_max(x,1)], 1)
        return output

class ReadoutLayer(Layer):
    """Graph Readout layer."""
    def __init__(self, placeholders,method='mean',**kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)
        self.ND=placeholders['nnodes']          

        self.method=method 
        self.istrain = placeholders['istrain'] 

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.method=="max":
            output=tf.reduce_max(x,1)
        elif self.method=="power":
            output=tf.reduce_sum(x*x,1)/self.ND
        elif self.method=="mean":
            output=tf.reduce_sum(x,1)/self.ND
        elif self.method=="sum":
            output=tf.reduce_sum(x,1)
        elif self.method=="meanmax":
            output= tf.concat([tf.reduce_max(x,1),tf.reduce_sum(x,1)/self.ND], 1)        
        
        output=tf.layers.batch_normalization(output,training=self.istrain) 
        
        return output
