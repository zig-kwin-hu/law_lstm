#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import random
def load_data_and_labels():
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	word_embeddings = []
	word2id = {}
	f = open("/data/disk1/private/lx/law/person.vec", "r")
	content = f.readline()
	while True:
		content = f.readline()
		if content == "":
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		word_embeddings.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	lists = [0.0 for i in range(len(word_embeddings[0]))]
	word_embeddings.append(lists)
	word_embeddings.append(lists)
	word_embeddings = np.array(word_embeddings, dtype=np.float32)

	relationhash = {}
	namehash = {}
	num_r = {}
	r_list = []

	x_train = []
	y_train = []
	nn = 0

	#fy = open('/data/disk1/private/lx/law/Re/train_y.txt','w')
	f = open("/data/disk1/private/lx/law/Fact_10w_macro_train", "r")
	content = f.readlines()
	random.shuffle(content)
	for i in content:
		z = i.strip().split("\t")
		if(len(z) != 2):
			continue
		x_train.append(z[0])
		y_train.append(z[1])
		for j in z[1].strip().split():
			if not j in relationhash:
				relationhash[j] = len(relationhash)
				namehash[relationhash[j]] = j
				num_r[relationhash[j]] = 0
			#fy.write(str(relationhash[j]) + ' ')
		#fy.write('\n')

	f.close()
	print len(x_train)
	x_test = []
	y_test = []
	la = 0
	#fy = open('/data/disk1/private/lx/law/Re/test_y.txt','w')
	f = open("/data/disk1/private/lx/law/Fact_10w_macro_test", "r")
	content = f.readlines()
	for i in content:
		z = i.strip().split("\t")
		if(len(z) != 2):
			continue
		x_test.append(z[0])
		y_test.append(z[1])
		for j in z[1].strip().split():
			if not j in relationhash:
				relationhash[j] = len(relationhash)
				namehash[relationhash[j]] = j
				num_r[relationhash[j]] = 0
			num_r[relationhash[j]] += 1
			#fy.write(str(relationhash[j]) + ' ')
		#fy.write('\n')


	f.close()
	print len(x_test)
	res = []
	fx = open('rel','w')
	for i in range(len(relationhash)):
		fx.write(str(num_r[i]) + '\n')
	#for i in relationhash:
	# 	r_list.append({'num':num_r[i],'id':i})
	# r_list.sort(key = lambda x:x['num'],reverse = True)
	# for i in r_list:
	# 	fx.write(i['id'] + ' ' + str(i['num']) + '\n')

	yz = []
	for i in xrange(0, len(y_test)):
		
		label = [0 for k in range(0, len(relationhash))]
		tmp = []
		for j in y_test[i].strip().split():
			try:
				uid = relationhash[j]
			except KeyError:
				continue 
			label[uid] = 1
			tmp.append(uid)
		yz.append(tmp)
		res.append(label)
	y_test = np.array(res)

	res = []
	for i in xrange(0, len(y_train)):
		label = [0 for k in range(0, len(relationhash))]
		for j in y_train[i].strip().split():
			try:
				uid = relationhash[j]
			except KeyError:
				continue
			label[uid] = 1
		res.append(label)
	y_train = np.array(res)

	max_document_length_train = sum([len(x.split()) for x in x_train])/len(x_train)
	max_document_length_test = max([len(x.split()) for x in x_test])
	max_document_length = max(max_document_length_train, max_document_length_test)
	print max_document_length_train, max_document_length_test
	max_document_length = 500
	# xlen = [len(x.split()) for x in x_train]
	# xlen.sort(
	# print xlen[int(len(x_train)*0.9)]
	size = len(x_train)
	size0 = 0
	size1 = 0

	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_train[i].split()
		for j in xrange(len(content)):
			if(j == max_document_length):
				break
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		x_train[i] = text
	x_train = np.array(x_train)

	size = len(x_test)
	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_test[i].split()
		for j in xrange(len(content)):
			if(j == max_document_length):
				break
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		x_test[i] = text
	x_test = np.array(x_test)
	print 'init finish'
	return word_embeddings, x_train, y_train, x_test, y_test, num_r, yz, namehash

# word_embeddings, x_train, y_train, x_test, y_test =  load_data_and_labels()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = (int)(round(len(data)/batch_size)) 
	for epoch in range(num_epochs):
 		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
