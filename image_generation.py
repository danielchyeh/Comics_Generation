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


   
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=1,
					   help='0 for training, 1 for testing')
parser.add_argument('--resume', type=bool, default=False,
					   help='True for resuming the model; False for initialization')
parser.add_argument('--z_dim', type=int, default=100,
					   help='Noise dimension')

parser.add_argument('--iter', type=int, default=10000,
					   help='number of training iter')
parser.add_argument('--batch_size', type=int, default=64,#256
					   help='Batch Size')
parser.add_argument('--display_step', type=int, default=20,
					   help='predict model on dev set after this many steps')
parser.add_argument('--dump_every', type=int, default=500,
					   help='predict model on dev set after this many steps')       
parser.add_argument('--checkpoint_every', type=int, default=2000,
					   help='Save model after this many steps')
  
parser.add_argument('--learn_rate', type=float, default=0.0001,
					   help='training learning rate')    

parser.add_argument('--model_dir', type=str, default="./model_gan/",#create a folder for the generated models
					   help='model direction')
parser.add_argument('--train_img_dir', type=str, default="./train_img/",
					   help='test image directory')
parser.add_argument('--img_dir', type=str, default="./samples/",#create a folder for the generated images
					   help='test image directory')
parser.add_argument('--train_dir', type=str, default="./MLDS_HW3_dataset/faces",#default folder from downloaded file
					   help='training data directory"')
parser.add_argument('--tag_path', type=str, default="./MLDS_HW3_dataset/tags_clean.csv",#default folder from downloaded file
					   help='training data tags')

parser.add_argument("test_path", type=str, help='sample test format')#argv[1]

parser.add_argument('--checkpoint_file', type=str, default="",
					   help='checkpoint_file to be load')
parser.add_argument('--prepro_dir', type=str, default="./prepro/",
					   help='tokenized train datas path')
parser.add_argument('--vocab', type=str, default="./vocab",
					   help='vocab processor path') 
parser.add_argument('--model', type=str, default="Improved_WGAN",
					   help='init model name')   
parser.add_argument('--pre_parameter', type=bool, default=True,
					   help='reload')     

args = parser.parse_args()



model_options = {
        'z_dim' : args.z_dim,
        'batch_size' : args.batch_size,
        'learn_rate' : args.learn_rate
    }

training_options = {
        'z_dim' : args.z_dim,
        'iter' : args.iter,
        'batch_size' : args.batch_size,
        'display_step' : args.display_step,
        'dump_every' : args.dump_every,
        'checkpoint_every' : args.checkpoint_every,
        'img_dir' : args.img_dir,
        'train_img_dir' : args.train_img_dir
    }



if not os.path.exists("./prepro/"):
    os.makedirs("./prepro/")
if args.pre_parameter == True:
    img_feat = cPickle.load(open(os.path.join(args.prepro_dir, "img_feat.dat"), 'rb'))
    tags_idx = cPickle.load(open(os.path.join(args.prepro_dir, "tag_ids.dat"), 'rb'))
    a_tags_idx = cPickle.load(open(os.path.join(args.prepro_dir, "a_tag_ids.dat"), 'rb'))
    k_tmp_vocab = cPickle.load(open(os.path.join(args.prepro_dir, "k_tmp_vocab_ids.dat"), 'rb'))
    vocab_processor = Vocab_Operator.restore(args.vocab)        

else:
    img_feat, tags_idx, a_tags_idx, vocab_processor, k_tmp_vocab = data_utils.load_train_data(args.train_dir,
    args.tag_path, args.prepro_dir, args.vocab)        


img_feat = np.array(img_feat, dtype='float32')/127.5 - 1.
test_tags_idx = data_utils.load_test(args.test_path, vocab_processor, k_tmp_vocab)

print("Image feature shape: {}".format(img_feat.shape))
print("Tags index shape: {}".format(tags_idx.shape))
print("Attribute Tags index shape: {}".format(a_tags_idx.shape))
print("Test Tags index shape: {}".format(test_tags_idx.shape))

data = Data(img_feat, tags_idx, a_tags_idx, test_tags_idx, args.z_dim, vocab_processor)



dcgan = dcgan.DCGAN(model_options, training_options, data, args.mode, args.resume, args.model_dir)

input_tensors, variables, loss, outputs, checks = dcgan.build_model()

if args.mode == 0: 
    dcgan.train(input_tensors, variables, loss, outputs, checks)
else:
    dcgan.test()


    