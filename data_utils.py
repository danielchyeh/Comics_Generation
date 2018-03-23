import numpy as np
import sys
import os
import csv
from scipy import misc
import collections
import _pickle as cPickle
from tensorflow.python.platform import gfile
import scipy.stats as stats
import math
import random
import copy


try:
	import cPickle as pickle
except ImportError:
	import pickle


vocab_threshold = 200
topk = 5


vocab = collections.defaultdict(int)
used_vocab = collections.defaultdict(int)
raw_c_tmps = []
attrib_a_tmps = [] 
img_feat = []
count = 0
count_eyes = 0
count_hair = 0
r_tmp_eh = []

a_tmp_idx = []
tmp_idx = []
train_path = "./MLDS_HW3_dataset/faces"
#tag_path = "./MLDS_HW3_dataset/tags_clean.csv"
#test_path = "./MLDS_HW3_dataset/sample_testing_text.txt"


class Data(object):
	def __init__(self, img_feat, tags_idx, a_tags_idx, test_tags_idx, z_dim, vocab_processor):
		self.z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
		self.length = len(tags_idx)
		self.current = 0
		self.img_feat = img_feat
		self.tags_idx = tags_idx
		self.a_tags_idx = a_tags_idx
		self.w_idx = np.arange(self.length)
		self.w_idx2 = np.arange(self.length)
		self.tmp = 0
		self.epoch = 0
		self.vocab_processor = vocab_processor
		self.vocab_size = len(vocab_processor._reverse_mapping)
		self.unk_id = vocab_processor._mapping['<UNK>']
		self.eos_id = vocab_processor._mapping['<EOS>']
		self.hair_id = vocab_processor._mapping['hair']
		self.eyes_id = vocab_processor._mapping['eyes']
		self.gen_info()
		self.test_tags_idx = self.gen_test_hot(test_tags_idx)
		self.fixed_z = self.next_noise_batch(len(self.test_tags_idx), z_dim)

		idx = np.random.permutation(np.arange(self.length))
		self.w_idx2 = self.w_idx2[idx]

	def gen_test_hot(self, test_intput):
		test_hot = []
		for tag in test_intput:
			eyes_hot = np.zeros([len(self.eyes_idx)])
			eyes_hot[np.where(self.eyes_idx == tag[2])[0]] = 1
			hair_hot = np.zeros([len(self.hair_idx)])
			hair_hot[np.where(self.hair_idx == tag[0])[0]] = 1
			tag_vec = np.concatenate((eyes_hot, hair_hot))
			test_hot.append(tag_vec)

		return np.array(test_hot)

	def gen_info(self):
		self.eyes_idx = np.array([idx for idx in set(self.a_tags_idx[:,0])])
		self.hair_idx = np.array([idx for idx in set(self.a_tags_idx[:,1])])
		self.type = []
		for a_tag in self.a_tags_idx:
			if a_tag[0] == self.unk_id:
				self.type.append(1)
			elif a_tag[1] == self.unk_id:
				self.type.append(2)
			else:
				self.type.append(0)
		self.type = np.array(self.type)

		self.one_hot = []
		for a_tag in self.a_tags_idx:
			eyes_hot = np.zeros([len(self.eyes_idx)])
			eyes_hot[np.where(self.eyes_idx == a_tag[0])[0]] = 1
			hair_hot = np.zeros([len(self.hair_idx)])
			hair_hot[np.where(self.hair_idx == a_tag[1])[0]] = 1
			tag_vec = np.concatenate((eyes_hot, hair_hot))
			self.one_hot.append(tag_vec)
		self.one_hot = np.array(self.one_hot)

	def next_data_batch(self, size, neg_sample=False):
		if self.current == 0:
			self.epoch += 1
			idx = np.random.permutation(np.arange(self.length))
			self.img_feat = self.img_feat[idx]
			self.tags_idx = self.tags_idx[idx]
			self.a_tags_idx = self.a_tags_idx[idx]
			self.type = self.type[idx]
			self.one_hot = self.one_hot[idx]
			idx = np.random.permutation(np.arange(self.length))
			self.w_idx = self.w_idx[idx]

		if self.current + size < self.length:
			img, tags, a_tags, d_t, widx, hot = self.img_feat[self.current:self.current+size], self.tags_idx[self.current:self.current+size], self.a_tags_idx[self.current:self.current+size], self.type[self.current:self.current+size], self.w_idx[self.current:self.current+size], self.one_hot[self.current:self.current+size]
			self.current += size

		else:
			img, tags, a_tags, d_t, widx, hot = self.img_feat[self.current:], self.tags_idx[self.current:], self.a_tags_idx[self.current:], self.type[self.current:], self.w_idx[self.current:], self.one_hot[self.current:]
			self.current = 0

		size = len(tags)
		type0_idx = np.where(d_t == 0)[0]
		if len(type0_idx) > 0:
			while True:
				mis_idx = np.where(np.mean(np.equal(a_tags[type0_idx], self.a_tags_idx[widx][type0_idx]), axis=1) == 1)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type0_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		# eye:unk, hair:tag
		type1_idx = np.where(d_t == 1)[0]
		if len(type1_idx) > 0:
			while True:
				mis_idx = np.where(np.equal(a_tags[type1_idx][:,1], self.a_tags_idx[widx][type1_idx,1]) == True)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type1_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		# eye:tag, hair:unk
		type2_idx = np.where(d_t == 2)[0]
		if len(type2_idx) > 0:
			while True:
				mis_idx = np.where(np.equal(a_tags[type2_idx][:,0], self.a_tags_idx[widx][type2_idx,0]) == True)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type2_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		return img, hot, a_tags, self.img_feat[widx], self.one_hot[widx]

	def next_noise_batch(self, size, dim):
		return self.z_sampler.rvs([size, dim]) #np.random.uniform(-1.0, 1.0, [size, dim])







class Vocab_Operator(object):
    def __init__(self, max_document_length, vocabulary, unknown_limit=float('Inf'), drop=False):
        self.max_document_length = max_document_length
        self._reverse_mapping = ['<UNK>', '<EOS>'] + vocabulary
        self.make_mapping()
        self.unknown_limit = unknown_limit
        self.drop = drop    
    
    def make_mapping(self):
        self._mapping = {}
        for i, vocab in enumerate(self._reverse_mapping):
            self._mapping[vocab] = i
            
    def trans(self, raw_documents, len_docu, vocab, const):
        
        a_array_idx = np.ones((len(raw_documents),len_docu), np.int32)
        for a in range(0,len(raw_documents),1):
        #for a in range(0,5,1):
            a_array_id = np.ones((1,len_docu), np.int32)
            attrib_a_tmps_div = raw_documents[a].split(' ')
            for b in range(0,len(attrib_a_tmps_div),1):
                if (attrib_a_tmps_div[b] == '<UNK>'):
        #            a_tmp_id.append(0)
                    a_array_id[0,b] = 0
                else:
                    for c in range(0,len(vocab),1):
                        if (attrib_a_tmps_div[b] == vocab[c]):
        #                    a_tmp_id.append(c + 2)
                            a_array_id[0,b] = c + const
        
        #    a_tmp_idx = np.array(a_tmp_idx)
            a_array_idx[a] = a_array_id
    
        return a_array_idx
    
    def save(self, filename):
        with gfile.Open(filename, 'wb') as f:
            f.write(pickle.dumps(self))
    @classmethod
    def restore(cls, filename):
        with gfile.Open(filename, 'rb') as f:
            return pickle.loads(f.read())
    

def load_train_data(img_dir, tag_path, prepro_dir, vocab_path, shuffle_time=1):

    img_feat = []
    with open(tag_path, 'r') as f:
        for ridx, row in enumerate(csv.reader(f)):
            tags = row[1].split('\t')
            for t in tags:
                tag = t.split(':')[0].strip()
                for w in tag.split():
                    vocab[w] += 1
    
        
    with open(tag_path, 'r') as f:
        for r_index, row in enumerate(csv.reader(f)):
            r_tmps = row[1].split('\t')
            c_tmp_eh = []
            k_tmp = {}#for value score
            r_tmp_eh = ['<UNK>','<UNK>']#[0] for eyes, [1] for hair
    #        if (int(row[0]) == 4):
            eye_flag = False
            hair_flag = False
            count_hair = 0
            count_eyes = 0
            
            for i in range(0,len(r_tmps),1):
                if r_tmps[i] != '':
                    r_tmp = r_tmps[i].split(':')[0].strip()
                    sc_value = r_tmps[i].split(':')[1].strip()
                    sc_value = int(sc_value)
                    r_tmp_sc = r_tmp.split()
                    for j in range(0,len(r_tmp_sc),1):
                        if (vocab[r_tmp_sc[j]] < vocab_threshold or len(r_tmp_sc) > 2
                            or r_tmp_sc[j] == 'long' or r_tmp_sc[j] == 'short'):
                            sc_value = -1
                        
                    if ((r_tmp.find('hair') > 0) and r_tmp.find(' ') > 0 and r_tmp.find('11') < 0):
                        r_tmp_dh = r_tmp.split(' ')
                        
                        if (((r_tmp_dh[0] == 'short') or (r_tmp_dh[0] == 'long') or (r_tmp_dh[0] == 'damage'))):#15611
                            r_tmp_dh = r_tmp_dh
                        else:
                            count_hair = count_hair + 1
                            if (count_hair > 1):#no more than one kind of hair
                                r_tmp_eh[0] = '<UNK>'
                                r_tmp_eh[1] = '<UNK>'
                                eye_flag = True
                                break
                            else:
                                r_tmp_eh[1] = r_tmp_dh[0]
                                sc_value = float('Inf')
                    
                    if(r_tmp.find('eyes') > 0 and r_tmp.find(' ') > 0 and r_tmp.find('11') < 0):
                        count_eyes = count_eyes + 1
                        if (count_eyes > 1):#no more than one kind of eyes
                            r_tmp_eh[0] = '<UNK>'
                            r_tmp_eh[1] = '<UNK>'
                            eye_flag = True
                            break 
                        else:                   
                            r_tmp_de = r_tmp.split(' ')
                            r_tmp_eh[0] = r_tmp_de[0]
                            sc_value = float('Inf')
                    
                    if (sc_value != -1):#remove values which is not useful
                        k_tmp[r_tmp] = sc_value
    
                        
            
            if (r_tmp_eh[0] == '<UNK>' and r_tmp_eh[1] == '<UNK>'):
                r_tmp_eh = r_tmp_eh
            else:
                a_tmp_eh = r_tmp_eh[0] + ' ' + r_tmp_eh[1]
    #            if (r_tmp_eh[0] == 'red' and r_tmp_eh[1] == 'red'):
    #                xx.append(r_index)
    #            if (r_tmp_eh[0] == '<UNK>' and r_tmp_eh[1] == 'blue'):
    #                xx1.append(r_index)
    #                if (r_index > 2000):
    #                    if (max(xx1) == (max(xx)+1) or max(xx1) == (max(xx)+2) or max(xx1) == (max(xx)+3) or max(xx1) == (max(xx)+4)):
    #                        xx2.append(r_index)
                    
                sor_k_tmp = sorted(k_tmp.items(), key=lambda x:x[1], reverse=True)
                for idx, (k, v) in enumerate(sor_k_tmp):
                    if idx < topk:
                        c_tmp_eh.append(k)
                        for w in k.split():
                            used_vocab[w] += 1
            
                c_tmp_eh = [r_tmp_eh[0] + ' eyes', r_tmp_eh[1] + ' hair']
            
                random.shuffle(c_tmp_eh)
                raw_c_tmps.append(' '.join(c_tmp_eh))
                
                random.shuffle(c_tmp_eh)
                raw_c_tmps.append(' '.join(c_tmp_eh))
                
                random.shuffle(c_tmp_eh)
                raw_c_tmps.append(' '.join(c_tmp_eh))
                
                random.shuffle(c_tmp_eh)
                raw_c_tmps.append(' '.join(c_tmp_eh))
                
                attrib_a_tmps.append(a_tmp_eh)
                attrib_a_tmps.append(a_tmp_eh)
                attrib_a_tmps.append(a_tmp_eh)
                attrib_a_tmps.append(a_tmp_eh)
                
                
                img_path = os.path.join(train_path, '{}.jpg'.format(r_index))
                feat = misc.imread(img_path)
                feat = misc.imresize(feat, [64, 64, 3])
                img_feat.append(feat)
                
                m_feat = np.fliplr(feat)
                img_feat.append(m_feat)
                
                feat_p5 = misc.imrotate(feat, 5)
                img_feat.append(feat_p5)
                
                feat_m5 = misc.imrotate(feat, -5)
                img_feat.append(feat_m5)
                
    img_feat = np.array(img_feat)
                    
    k_tmp_vocab = []
    sor_used_vocab = sorted(used_vocab.items(), key=lambda x:x[1], reverse=True)
    for k, v in sor_used_vocab:
    	k_tmp_vocab.append(k)            
    
    
    max_length = max([len(tags.split()) for tags in raw_c_tmps])
    vocab_oper = Vocab_Operator(max_document_length=max_length, vocabulary=k_tmp_vocab)
    
    a_tmp_idx = vocab_oper.trans(attrib_a_tmps, 2, k_tmp_vocab, 2)
    train_tmp_idx = vocab_oper.trans(raw_c_tmps, 4, k_tmp_vocab, 2)
    
    cPickle.dump(img_feat, open(os.path.join(prepro_dir, "img_feat.dat"), 'wb'))
    cPickle.dump(train_tmp_idx, open(os.path.join(prepro_dir, "tag_ids.dat"), 'wb'))
    cPickle.dump(a_tmp_idx, open(os.path.join(prepro_dir, "a_tag_ids.dat"), 'wb'))

    cPickle.dump(k_tmp_vocab, open(os.path.join(prepro_dir, "k_tmp_vocab_ids.dat"), 'wb'))
    
    vocab_oper.save(vocab_path)
    
    return img_feat, train_tmp_idx, a_tmp_idx, vocab_oper, k_tmp_vocab

def load_test(test_path, vocab_oper, k_tmp_vocab):
    test = []
    testline2 = []
    with open(test_path, 'r') as f:
    	for line in f.readlines():
    		line = line.strip().split(',')[1]
    		test.append(line)
            
    for tes in test:
        te = tes.split(' ')
        te_size = len(te)
    
        if te_size == 4:
            if te[1] == 'eyes':
                te_exchange = []
                te_exchange.append(te[2])
                te_exchange.append(te[3])
                te_exchange.append(te[0])
                te_exchange.append(te[1])
                
                teh = te_exchange
            else:
                teh = te
                
        else:
            if te[1] == 'eyes':
                te_exchagev1 = []
                te_exchagev1.append('blue')
                te_exchagev1.append('hair')
                te_exchagev1.append(te[0])
                te_exchagev1.append(te[1])            
                teh = te_exchagev1
                
            if te[1] == 'hair':
                te_exchagev2 = []
                te_exchagev2.append(te[0])
                te_exchagev2.append(te[1])
                te_exchagev2.append('green')
                te_exchagev2.append('eyes')           
                teh = te_exchagev2
                
        stringv1 = ''
        for t in teh:
            stringv1 = stringv1 + t + ' '
        stringv2 = stringv1[:-1]    
        
        testline2.append(stringv2)
    
            
    test_tmp_idx = vocab_oper.trans(testline2, 4, k_tmp_vocab, 2) 
    
    return test_tmp_idx



def train_dump_img(train_img_dir, img_feats, iters):
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
	
    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        path = os.path.join(train_img_dir, 'iters_{}_train_{}.jpg'.format(iters, idx))
        misc.imsave(path, img_feat)
#        path = os.path.join(img_dir, 'sample_{}_{}.jpg'.format(idx+1, num_id))
#        misc.imsave(path, img_feat)


def test_dump_img(img_dir, img_feats, iters, num_id):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
	
    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
#        path = os.path.join(img_dir, 'iters_{}_test_{}.jpg'.format(iters, idx))
#        misc.imsave(path, img_feat)
        path = os.path.join(img_dir, 'sample_{}_{}.jpg'.format(idx+1, num_id))
        misc.imsave(path, img_feat)

#a_tmp_idx = np.ones((len(attrib_a_tmps),2), np.int32)
#for a in range(0,len(attrib_a_tmps),1):
##for a in range(0,5,1):
#    a_tmp_id = np.ones((1,2), np.int32)
#    attrib_a_tmps_div = attrib_a_tmps[a].split(' ')
#    for b in range(0,len(attrib_a_tmps_div),1):
#        if (attrib_a_tmps_div[b] == '<UNK>'):
##            a_tmp_id.append(0)
#            a_tmp_id[0,b] = 0
#        else:
#            for c in range(0,len(k_tmp_vocab),1):
#                if (attrib_a_tmps_div[b] == k_tmp_vocab[c]):
##                    a_tmp_id.append(c + 2)
#                    a_tmp_id[0,b] = c + 2
#
##    a_tmp_idx = np.array(a_tmp_idx)
#    a_tmp_idx[a] = a_tmp_id
##    for d in range(0,len(attrib_a_tmps),1):
##    a_tmp_idx = [a_tmp_id[0,0], a_tmp_id[0,1]]
##
##    a_tmp_idx.append(a_tmp_id)        
##    a_tmp_idx = np.array(a_tmp_idx)




