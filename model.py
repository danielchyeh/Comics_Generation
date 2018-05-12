import tensorflow as tf
import tensorflow.contrib as tc

slim = tf.contrib.slim
layers = tf.contrib.layers


class Generator(object):
    def __init__(self, 
        max_seq_length, 
        img_row, 
        img_col,
        train=True):

        self.max_seq_length = max_seq_length
        self.img_row = img_row
        self.img_col = img_col
        self.train = train
		
    def __call__(self, seq_idx, z, reuse=False, train=True):
        
        depthsG = [512, 256, 128, 64] #filters for G
        s_size = 4  
        self.depthsG = depthsG + [3]#the last output channel will be 3 
        self.s_size = s_size

        tags_vectors = seq_idx

        with tf.variable_scope('g_net', reuse=reuse):
            with slim.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                ):
                noise_vector = tf.concat([z, tags_vectors], axis=-1)
                
                outputs = tf.layers.batch_normalization(layers.fully_connected(noise_vector, 4 * 4 * self.depthsG[0]), \
                training=train, momentum=0.9, epsilon=1e-5)
                
                outputs = tf.reshape(outputs, [-1, 4, 4, self.depthsG[0]])
                
                outputs = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(outputs, self.depthsG[1], [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                
                outputs = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(outputs, self.depthsG[2], [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                
                outputs = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(outputs, self.depthsG[3], [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                
                outputs = layers.conv2d_transpose(outputs, self.depthsG[4], [5, 5], stride=2, normalizer_fn=None, activation_fn=None) 
                
                outputs = tf.nn.tanh(outputs)
            
        return outputs
            

class Discriminator(object):
    
    
    def __init__(self, 
        max_seq_length, 
        img_row,
        img_col):

        self.max_seq_length = max_seq_length
        self.img_row = img_row
        self.img_col = img_col
		
    def __call__(self, seq_idx, inputs, reuse=False):
        
        def lrelu(x, leak=0.2, name="lrelu"):
            with tf.variable_scope(name):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)
        
        
        depthsD = [64, 128, 256, 512] 
        self.depthsD = [1] + depthsD 

        with tf.variable_scope('d_net', reuse=reuse):
            with slim.arg_scope(
                [layers.conv2d, layers.fully_connected],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
                ):
                
                outputs = lrelu(layers.conv2d(inputs, self.depthsD[1], [5, 5], stride=2))
                
                outputs  = lrelu(layers.conv2d(outputs, self.depthsD[2], [5, 5], stride=2))
                
                outputs  = lrelu(layers.conv2d(outputs, self.depthsD[3], [5, 5], stride=2))
               
                outputs  = lrelu(layers.conv2d(outputs, self.depthsD[4], [5, 5], stride=2))
                
                
                tiled_embeddings = tf.tile(tf.expand_dims(tf.expand_dims(seq_idx, 1), 2), [1, 4, 4, 1])

                h3_concat = tf.concat([outputs, tiled_embeddings], axis=-1)

                net = lrelu(layers.conv2d(h3_concat, self.depthsD[4], [1, 1], stride=1, padding='valid'))
                net = layers.flatten(net)
                # net = layers.fully_connected(net, 4*4*512)
                net = layers.fully_connected(net, self.depthsD[0], normalizer_fn=None, activation_fn=None)
                
        return net

            