import tensorflow as tf
import tensorflow.contrib as tc


class Generator(object):
    def __init__(self, 
        max_seq_length, 
        img_row, 
        img_col):

        self.max_seq_length = max_seq_length
        self.img_row = img_row
        self.img_col = img_col
		
    def __call__(self, seq_idx, z, reuse=False, train=True):
        
        depths = [256, 128, 64, 32] 
        s_size = 4  
        self.depths = depths + [3]
        self.s_size = s_size

        tags_vectors = seq_idx

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()

            noise_vector = tf.concat([tags_vectors, z], axis=1)

            with tf.variable_scope('reshape'):            
                outputs = tc.layers.fully_connected(
                noise_vector, 4*4*self.depths[0],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
                outputs = tf.nn.relu(tf.reshape(tf.layers.batch_normalization(outputs, training=train), [-1, 4, 4, self.depths[0]]))
            
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]

class Discriminator(object):
    
    
    def __init__(self, 
        max_seq_length, 
        img_row,
        img_col):

        self.max_seq_length = max_seq_length
        self.img_row = img_row
        self.img_col = img_col
		
    def __call__(self, seq_idx, inputs, reuse=True):
        
        def leaky_relu(x, alpha=0.2, name=''):
            return tf.maximum(tf.minimum(0.0, alpha * x), x, name=name)
        
        depths = [32, 64, 128, 128] 
        self.depths = [1] + depths 

        tags_vectors = seq_idx

        with tf.variable_scope("d_net") as scope:

            if reuse == True:
                scope.reuse_variables()

            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(inputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
            with tf.variable_scope('concat'): 
                tags_vectors = tf.tile(tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2), [1, 8, 8, 1])
    
                cond_outputs = tf.concat([outputs, tags_vectors], axis=-1)
                
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(cond_outputs, self.depths[4], [1, 1], strides=(1, 1), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=True), name='outputs')
            with tf.variable_scope('classify'):
                outputs = tf.squeeze(tf.layers.conv2d(outputs, self.depths[0], [8, 8], strides=(1, 1), padding='valid'), [1, 2, 3])
    
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs
    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]



class Sampler(object):
    def __init__(self, 
        max_seq_length,  
        img_row, 
        img_col):

        self.max_seq_length = max_seq_length
        self.img_row = img_row
        self.img_col = img_col
		
    def __call__(self, seq_idx, z, reuse=True, train=False):
        
        depths = [256, 128, 64, 32] 
        s_size = 4  
        self.depths = depths + [3]
        self.s_size = s_size

        tags_vectors = seq_idx

        with tf.variable_scope("g_net") as scope:

            if reuse:
                scope.reuse_variables()

            noise_vector = tf.concat([tags_vectors, z], axis=1)

            with tf.variable_scope('reshape'):            
                outputs = tc.layers.fully_connected(
                noise_vector, 4*4*self.depths[0],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
                outputs = tf.nn.relu(tf.reshape(tf.layers.batch_normalization(outputs, training=train), [-1, 4, 4, self.depths[0]]))
            
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=train), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]


