from layers_tf import *
from metrics_tf import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.entropy = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.update_ops=None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._entropy()

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
    def _entropy(self):
        self.entropy= masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
    def _entropy(self):
        self.entropy= masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class DSGCNN(Model):
    def __init__(self, placeholders, input_dim, readout='mean', nkernel=1, **kwargs):
        super(DSGCNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim        
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders        
        self.nkernel=nkernel
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.readout=readout
        self.build()

    def _loss(self):
        # Weight decay loss
        self.loss=0
        tmp=FLAGS.weight_decay
        if tmp>0:
            for layers in self.layers:
                for var in layers.vars.values():            
                    self.loss += tmp * tf.nn.l2_loss(var) 

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])
                                                   
    def _entropy(self):
        self.entropy= softmax_cross_entropy(self.outputs, self.placeholders['labels'])
                                                  

    def _accuracy(self):
        self.accuracy = inductive_multiaccuracy(self.outputs, self.placeholders['labels'])

    def _build(self):

        self.layers.append(GraphConvolutionBatch(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,                                            
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            nkernel=self.nkernel,
                                            bias=True,
                                            dropout=False,                                            
                                            logging=self.logging))
                                            
        self.layers.append(GraphConvolutionBatch(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,                                            
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            nkernel=self.nkernel,
                                            bias=True,
                                            dropout=True,                                            
                                            logging=self.logging))  

        self.layers.append(GraphConvolutionBatch(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,                                            
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            nkernel=self.nkernel,
                                            bias=True,
                                            dropout=True,                                            
                                            logging=self.logging))   

        self.layers.append(ReadoutLayer(method=self.readout,placeholders=self.placeholders))

        self.layers.append(Dense(input_dim=FLAGS.dense,
                                output_dim=FLAGS.dense2,
                                placeholders=self.placeholders,
                                act=tf.nn.relu, 
                                dropout=True,
                                bias=True,
                                logging=self.logging)) 

        self.layers.append(Dense(input_dim=FLAGS.dense2,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=False,
                                 logging=self.logging))       

    def predict(self):
        return tf.nn.softmax(self.outputs)



class DSSGCN_GC_BATCH(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(DSSGCN_GC_BATCH, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim        
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)        
        self.build()

    def _loss(self):
        # Weight decay loss
        tmp=FLAGS.weight_decay
        tmpcoef=FLAGS.weight_decay

        for i in range(0,len(self.layers)):
            for nm in list(self.layers[i].vars.keys()):
                if nm =='bias' :
                    continue
                if nm[0]=='s':
                    self.loss += tmpcoef * tf.nn.l2_loss(self.layers[i].vars[nm])
                else:
                    self.loss += tmp * tf.nn.l2_loss(self.layers[i].vars[nm])        
        
        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])
       
    def _entropy(self):
        self.entropy= softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = inductive_multiaccuracy(self.outputs, self.placeholders['labels'])

    def _build(self):

        inpdim=self.input_dim

        for i in range(0,len(FLAGS.hidden)):
            if FLAGS.hidden[i]=='mean' or FLAGS.hidden[i]=='max' or FLAGS.hidden[i]=='meanmax':
                self.layers.append(AggLayer(method=FLAGS.hidden[i],placeholders=self.placeholders))
                if FLAGS.hidden[i]=='meanmax':
                    inpdim*=2
            elif FLAGS.hidden[i]=='pool':
                self.layers.append(PoolLayer(placeholders=self.placeholders))
            elif FLAGS.hidden[i] > 0:
                self.layers.append(GraphConvolutionwithDephSepBatch(input_dim=inpdim,
                                                    output_dim=FLAGS.hidden[i],
                                                    placeholders=self.placeholders,                                            
                                                    act=FLAGS.activation_funcs[i], 
                                                    bias=FLAGS.biases[i],                                                    
                                                    dropout=(FLAGS.isdroput_inp[i], FLAGS.isdroput_kernel[i]),                                             
                                                    logging=self.logging))
                inpdim=np.abs(FLAGS.hidden[i])
            elif FLAGS.hidden[i] < 0:
                self.layers.append(Dense(input_dim=inpdim,
                                 output_dim=np.abs(FLAGS.hidden[i]),
                                 placeholders=self.placeholders,
                                 act=FLAGS.activation_funcs[i],
                                 dropout=FLAGS.isdroput_inp[i],
                                 bias=FLAGS.biases[i],
                                 logging=self.logging)) 
                inpdim=np.abs(FLAGS.hidden[i])
                  
                

    def predict(self):
        return tf.nn.sigmoid(self.outputs)

