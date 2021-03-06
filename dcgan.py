import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import os
import time
from model import Generator, Discriminator
import data_utils
from scipy import misc


class DCGAN:    
    def __init__(self, m_options, t_options, data, mode, resume, model_dir):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.m_options = m_options
        self.t_options = t_options
        self.data = data
        self.img_row = self.data.img_feat.shape[1]
        self.img_col = self.data.img_feat.shape[2]
        self.d_epoch = 1
        self.mode = mode
        self.resume = resume
        self.model_dir = model_dir
        

        
        
    def build_model(self):  
    
        self.g_net = Generator( 
		max_seq_length=self.data.tags_idx.shape[1], 
		img_row=self.img_row,
		img_col=self.img_col,
        	train=True)

        self.d_net = Discriminator( 
		max_seq_length=self.data.tags_idx.shape[1], 
		img_row=self.img_row,
		img_col=self.img_col)
        

        self.t_real_image = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="img")
        self.t_wrong_image = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="w_img")               
        self.t_real_caption = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="seq")
        self.t_wrong_caption = tf.placeholder(tf.float32, [None, len(self.data.eyes_idx)+len(self.data.hair_idx)], name="w_seq")
        self.t_z = tf.placeholder(tf.float32, [None, self.m_options['z_dim']])


        self.fake_image = self.g_net(self.t_real_caption, self.t_z, train=True)


        self.d_1 = self.d_net(self.t_real_caption, self.fake_image) # f img, r text
        self.d = self.d_net(self.t_real_caption, self.t_real_image, reuse=True) # r img, r text
        self.d_2 = self.d_net(self.t_wrong_caption, self.t_real_image, reuse=True) # r img, w text
        self.d_3 = self.d_net(self.t_real_caption, self.t_wrong_image, reuse=True) # w img, r text
        
        self.sampler = tf.identity(self.g_net(self.t_real_caption, self.t_z, reuse=True, train=False), name='sampler')
        
        

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.ones_like(self.d_1))) 

        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
			+ (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.zeros_like(self.d_1))) + \
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_2, labels=tf.zeros_like(self.d_2))) +\
			tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_3, labels=tf.zeros_like(self.d_3))) ) / 3 
        
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd_net')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g_net')
        
        
        self.global_step = tf.Variable(0, name='g_global_step', trainable=False)
        
        self.d_updates = tf.train.AdamOptimizer(self.m_options['learn_rate'], 0.5, 0.9).minimize(loss=self.d_loss, var_list=self.d_vars) #, epsilon=1e-08, decay=0.0

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.g_updates  = tf.train.AdamOptimizer(self.m_options['learn_rate'], 0.5, 0.9).minimize(loss=self.g_loss, var_list=self.g_vars, global_step=self.global_step)
        
        
        

#        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        #Choose to resume or initialize
        if self.resume or self.mode == 1:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        else:
            self.sess.run(tf.global_variables_initializer())
            
        
        
        input_tensors = {'t_real_image' : self.t_real_image,'t_wrong_image' : self.t_wrong_image,
                         't_real_caption' : self.t_real_caption,'t_z' : self.t_z, 't_wrong_caption' : self.t_wrong_caption}

        variables = {'d_vars' : self.d_vars,'g_vars' : self.g_vars}


        loss = {'g_loss' : self.g_loss,'d_loss' : self.d_loss}

        outputs = {'generator' : self.fake_image, 'sampler' : self.sampler,
                   'd_update' :self.d_updates, 'g_update' :self.g_updates,
                   'g_step' :self.global_step, 'saver' :self.saver}#, 'check_prefix':self.checkpoint_prefix}

        checks = {
                'd_loss1': self.d,
                'd_loss2': self.d_1,
                'd_loss3' : self.d_2,
                'd_loss4' : self.d_3,
                'sess' :self.sess
                }
		
        return input_tensors, variables, loss, outputs, checks
    
    
    def test(self): 
        """
        print("Start testing DCGAN...\n")
        current_step = tf.train.global_step(self.sess, self.global_step)
        
        for num_pic in range(0,5,1):
            z = self.data.fixed_z
            feed_dict = {
                    self.t_real_caption:self.data.test_tags_idx,
        			     self.t_z:z
                }
                                       
            f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
        
            data_utils.test_dump_img(self.t_options['img_dir'], f_imgs, current_step, num_pic+1)
                                    
        print("Dump test image")  
        """
        
        print("Start testing DCGAN...\n")
        #current_step = tf.train.global_step(self.sess, self.global_step)
        
        for j in range(len(self.data.test_tags_idx)) :
            #batch_tag = np.tile(self.data.test_tags_idx[j], (4,1))
            #print("batch tag shape: {}".format(batch_tag.shape))
            
            #z = self.data.fixed_z
            
            batch_y = np.tile(self.data.test_tags_idx[j], (5,1))
            noise = np.random.uniform(-1, 1, [batch_y.shape[0], 100]) 
            
            f_imgs = self.sess.run(self.sampler, feed_dict={self.t_real_caption:batch_y, self.t_z:noise})
            
            for i in range(5):
                img_feats = (f_imgs[i] + 1.)/2 * 255.
                img_feats = np.array(img_feats, dtype=np.uint8)
                
                path = os.path.join(self.t_options['img_dir'], 'sample_{}_{}.jpg'.format(j+1, i+1))
                misc.imsave(path, img_feats)
        
        print("Dump test image")  
        
        
    
    def train(self, input_tensors, variables, loss, outputs, checks):  
        
        print("Start training DCGAN...\n")
        
        for t in range(self.t_options['iter']):
        
            d_cost = 0
        
            for d_ep in range(self.d_epoch):
        
                img, tags, _, w_img, w_tags = self.data.next_data_batch(self.t_options['batch_size'])
                z = self.data.next_noise_batch(len(tags), self.t_options['z_dim'])
        
                feed_dict = {
                        input_tensors['t_real_caption']:tags,
                        input_tensors['t_real_image']:img,    
                        input_tensors['t_z']:z,
                        input_tensors['t_wrong_caption']:w_tags,
                        input_tensors['t_wrong_image']:w_img
                        }
        
                loss, _ = self.sess.run([self.d_loss, self.d_updates], feed_dict=feed_dict)
        
                d_cost += loss/self.d_epoch
                
                
                
        
            z = self.data.next_noise_batch(len(tags), self.t_options['z_dim'])
            feed_dict = {
                    input_tensors['t_real_image']:img,
                    input_tensors['t_wrong_caption']:w_tags,
                    input_tensors['t_wrong_image']:w_img,
                    input_tensors['t_real_caption']:tags,
                    input_tensors['t_z']:z
                    }
        
            #_, loss, step = self.sess.run([self.g_updates, self.g_loss, self.global_step], feed_dict=feed_dict)
            generated, loss, _, step = self.sess.run([outputs['generator'], self.g_loss, self.g_updates, self.global_step], feed_dict=feed_dict)
            #_, loss = sess.run([outputs['generator'], g_updates, loss['g_loss']], feed_dict=feed_dict)
            current_step = tf.train.global_step(self.sess, self.global_step)
        
            g_cost = loss
        
            if current_step % self.t_options['display_step'] == 0:
                print("Epoch {}, Current_step {}".format(self.data.epoch, current_step))
                print("Discriminator loss :{}".format(d_cost))
                print("Generator loss     :{}".format(g_cost))
                print("---------------------------------")
        
            if current_step % self.t_options['checkpoint_every'] == 0:              
                self.saver.save(self.sess, self.model_dir+"gan_{}.ckpt".format(current_step))
        
            if current_step % self.t_options['dump_every'] == 0:

                z = self.data.fixed_z
                feed_dict = {
                    self.t_real_caption:self.data.test_tags_idx,
        			     self.t_z:z
                }
                                       
                f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
        
                data_utils.train_dump_img(self.t_options['train_img_dir'], f_imgs, current_step)
                                   
                print("Dump train image")           
      
    
