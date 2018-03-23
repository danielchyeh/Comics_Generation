import tensorflow as tf
import numpy as np

import argparse
#import pickle
from os.path import join
import scipy.misc
import time
import os
import sys
#import shutil
import data_utils
import dcgan
from data_utils import Data
from data_utils import Vocab_Operator

import _pickle as cPickle


mode = 1
resume = False
z_dim = 100
iterr = 1000000
batch_size = 256
display_step = 20
display_every = 500
dump_every = 500
checkpoint_every = 2000
learn_rate = 2e-4
model_dir = "./model_gan/"
train_img_dir = "./train_img/"
img_dir = "./samples/"
#train_dir = "./MLDS_HW3_dataset/faces"
#tag_path = "./MLDS_HW3_dataset/tags_clean.csv"

test_path = sys.argv[1]
checkpoint_file = ""
prepro_dir = "./prepro/"
vocab = "./vocab"
pre_parameter = True
model = "Improved_WGAN"
   


model_options = {
        'z_dim' : z_dim,
        'batch_size' : batch_size,
        'learn_rate' : learn_rate
    }

training_options = {
        'z_dim' : z_dim,
        'iter' : iterr,
        'batch_size' : batch_size,
        'display_step' : display_step,
        'dump_every' : dump_every,
        'checkpoint_every' : checkpoint_every,
        'img_dir' : img_dir,
        'train_img_dir' : train_img_dir
    }



if not os.path.exists("./prepro/"):
    os.makedirs("./prepro/")
if pre_parameter == True:
    img_feat = cPickle.load(open(os.path.join(prepro_dir, "img_feat.dat"), 'rb'))
    tags_idx = cPickle.load(open(os.path.join(prepro_dir, "tag_ids.dat"), 'rb'))
    a_tags_idx = cPickle.load(open(os.path.join(prepro_dir, "a_tag_ids.dat"), 'rb'))
    k_tmp_vocab = cPickle.load(open(os.path.join(prepro_dir, "k_tmp_vocab_ids.dat"), 'rb'))
    vocab_processor = Vocab_Operator.restore(vocab)        

else:
    img_feat, tags_idx, a_tags_idx, vocab_processor, k_tmp_vocab = data_utils.load_train_data(train_dir,
    tag_path, prepro_dir, vocab)        


img_feat = np.array(img_feat, dtype='float32')/127.5 - 1.
test_tags_idx = data_utils.load_test(test_path, vocab_processor, k_tmp_vocab)

print("Image feature shape: {}".format(img_feat.shape))
print("Tags index shape: {}".format(tags_idx.shape))
print("Attribute Tags index shape: {}".format(a_tags_idx.shape))
print("Test Tags index shape: {}".format(test_tags_idx.shape))

data = Data(img_feat, tags_idx, a_tags_idx, test_tags_idx, z_dim, vocab_processor)



dcgan = dcgan.DCGAN(model_options, training_options, data, mode, resume, model_dir)

input_tensors, variables, loss, outputs, checks = dcgan.build_model()

if mode == 0: 
    dcgan.train(input_tensors, variables, loss, outputs, checks)
else:
    dcgan.test()


    
