#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from init import *

class LSTM_MODEL(object):

	def __init__(self, word_embeddings, config):

		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size
		num_classes = config.num_classes
		hits_k = config.hits_k
		self.input_x = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes])
		self.keep_prob = tf.placeholder(tf.float32)

		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 0.0)

		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

		self.initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device("/cpu:0"), tf.name_scope("embedding"):
			#embedding = tf.get_variable("embedding", [vocab_size, size])
			embedding = tf.Variable(word_embeddings, trainable = False)
			inputs = tf.nn.embedding_lookup(embedding, self.input_x)

		#inputs = tf.nn.dropout(inputs, config.keep_prob)
		# outputs = []
		# state = self._initial_state
		# with tf.variable_scope("RNN"):
		# 	for time_step in range(num_steps):
		# 		if time_step > 0: tf.get_variable_scope().reuse_variables()
		# 		(cell_output, state) = cell(inputs[:, time_step, :], state)
		# 		outputs.append(cell_output)
		outputs,_ = tf.nn.dynamic_rnn(cell,inputs,initial_state = self.initial_state)


		output = tf.expand_dims(tf.reshape(tf.concat(outputs,1), [batch_size, -1, size]), -1)

		with tf.name_scope("maxpool"):
			output_pooling = tf.nn.max_pool(output,
					ksize=[1, num_steps, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
			self.output = tf.reshape(output_pooling, [-1, size])

		with tf.name_scope("output"):
			softmax_w = tf.get_variable("softmax_w", [size, num_classes])
			softmax_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="softmax_b")
			self.scores = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.name_scope("loss"):
			self.soft = tf.nn.softmax(self.scores)
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores,labels =  self.input_y)
			self.loss = tf.reduce_mean(losses)

		with tf.name_scope("accuracy"):
			self.sums = tf.reduce_sum(self.input_y)
			self.hits = []
			for k in hits_k:
				topk_indices = tf.nn.top_k(self.scores, k=k).indices
				pred = tf.reduce_sum(tf.one_hot(topk_indices, num_classes), 1)
				self.hits.append(tf.reduce_sum(tf.multiply(pred, self.input_y)))
			self.sorted_indices = tf.nn.top_k(self.scores, k=num_classes).indices
			self.topk_sorted = tf.nn.top_k(self.scores, k=hits_k[-1]).indices

		self.initop = tf.initialize_all_variables()
		self.saver = tf.train.Saver()

		# if not is_training:
		# 	return
		# optimizer = tf.train.AdamOptimizer(0.001)
		# grads_and_vars = optimizer.compute_gradients(cnn.loss)
		# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

class Config(object):

	def __init__(self):
		self.num_layers = 2
		self.batch_size = 32
		self.keep_prob = 0.5
		self.num_epochs = 20
		self.num_steps = 50
		self.hidden_size = 50
		self.vocab_size = 10000
		self.num_classes = 550
		self.hits_k = [1]

def main(_):
	w, x_train, y_train, x_dev, y_dev, sz, answer, namehash = load_data_and_labels()
	config = Config()
	config.num_steps = len(x_train[0])
	config.hidden_size = len(w[0])
	config.vocab_size = len(w)
	config.num_classes = len(y_train[0])
	
	eval_config = Config()
	eval_config.num_steps = len(x_train[0])
	eval_config.hidden_size = len(w[0])
	eval_config.vocab_size = len(w)
	eval_config.num_classes = len(y_train[0])
	eval_config.keep_prob = 1.0

	# eval_config.batch_size = 64
	model_path = '/data/disk1/private/lx/law'+'/lstmmodel1/model'
#	print model_path
	zero_x = [0 for i in range(config.num_steps)]
	zero_y = [0 for i in range(config.num_classes)]

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = LSTM_MODEL(word_embeddings = w, config = config)

			lr = tf.Variable(0.001, trainable=False)
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(lr)
			train_op = optimizer.minimize(m.loss,global_step = global_step)
			sess.run(tf.initialize_all_variables())
			#load_path = m.saver.restore(sess, model_path + '-1000')  

			def Outout_Re(x_batch):

				feed_dict = {
					m.input_x: x_batch,
					m.keep_prob: 1.0
				}
				output = sess.run([m.output], feed_dict=feed_dict)
				return output

			def train_step(x_batch, y_batch):
				"""
				A single training step
				"""
				feed_dict = {
					m.input_x: x_batch,
					m.input_y: y_batch,
					m.keep_prob: 0.5
				}
				_, step, loss, sc = sess.run(
					[train_op, global_step, m.loss, m.soft], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				if step % 500 == 0:
					#print sc
	 				print("{}: step {}, loss {:g}".format(time_str, step, loss))
					path = m.saver.save(sess, model_path, global_step = step)

	 				

			def dev_step(x_batch, y_batch, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
					m.input_x: x_batch,
					m.input_y: y_batch,
					m.keep_prob: 1.0
				}
				cur_hits, cur_sum, predict, loss = sess.run([m.hits, m.sums, m.predictions 	, m.loss], feed_dict=feed_dict)
				# print sc

				return cur_hits, cur_sum, predict, loss

			def output():
				num = (int)(len(y_train) / (float)(eval_config.batch_size))

				fx = open('/data/disk1/private/lx/law/Re/train_x.txt','w')
				for i in range(num):
					output = Outout_Re(x_train[i * eval_config.batch_size:(i+1)*eval_config.batch_size])
					for k in output[0]:
						for j in k:
							fx.write(str(j) + ' ')
						fx.write('\n')

				fx = open('/data/disk1/private/lx/law/Re/test_x.txt','w')
					
				num = (int)(len(y_dev) / (float)(eval_config.batch_size))
				# print num
					
				for i in range(num):
					output = Outout_Re(x_dev[i * eval_config.batch_size:(i+1)*eval_config.batch_size])
					for k in output[0]:
						for j in k:
							fx.write(str(j) + ' ')
						fx.write('\n')

				left = len(y_dev) - num * eval_config.batch_size
				output = Outout_Re(np.append(x_dev[num * eval_config.batch_size:], [zero_x for i in range(eval_config.batch_size - left)], axis = 0))

				for k in range(left):
					for j in output[0][k]:
						fx.write(str(j) + ' ')
					fx.write('\n')


			batches = batch_iter(list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
			print 'prepare finish'
			k = 0
			print 'output finish'

			for batch in batches:
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				current_step = tf.train.global_step(sess, global_step)

				if current_step % 1000 == 0:

					print("\nEvaluation:")
					tp = [0]*config.num_classes
					fp = [0]*config.num_classes

					tn = datetime.datetime.now()
					hits = [0]*len(config.hits_k)
					p = [0]*len(config.hits_k)
				 	r = [0]*len(config.hits_k)
					p_indice = [float(config.hits_k[i]) for i in range(len(hits))]
					all_count = 0.0
					losses = 0.0
					num = (int)(len(y_dev) / (float)(eval_config.batch_size))
					print num
					
					for i in range(num):
						cur_hits, cur_sum, predict, cur_loss = dev_step(x_dev[i * eval_config.batch_size:(i+1)*eval_config.batch_size], y_dev[i * eval_config.batch_size:(i+1)*eval_config.batch_size])
						hits = list(map(lambda x: x[0]+x[1], zip(hits, cur_hits)))
						p_value = [eval_config.batch_size*indice for indice in p_indice]
						p = list(map(lambda x: x[0]+x[1], zip(p, p_value)))

						losses += cur_loss
						all_count += cur_sum

						for j in range(eval_config.batch_size):
							sign = 0
							#print j,predict[j]
							for k in answer[i * eval_config.batch_size + j]:
								#print k
								if(k == predict[j]):
									tp[k] += 1
									sign = 1
							if(sign == 0):
								fp[predict[j]] += 1
							
					left = len(y_dev) - num * eval_config.batch_size
					cur_hits, cur_sum, predict, cur_loss = dev_step(np.append(x_dev[num * eval_config.batch_size:], [zero_x for i in range(eval_config.batch_size - left)], axis = 0), np.append(y_dev[num * eval_config.batch_size:], [zero_y for i in range(eval_config.batch_size - left)], axis = 0))
					
					hits = list(map(lambda x: x[0]+x[1], zip(hits, cur_hits)))
					p_value = [left * indice for indice in p_indice]
					p = list(map(lambda x: x[0]+x[1], zip(p, p_value)))

					losses += cur_loss
					all_count += cur_sum

					for j in range(left):
						sign = 0
						for k in answer[num * eval_config.batch_size + j]:
							if(k == predict[j]):
								tp[k] += 1
								sign = 1
						if(sign == 0):
							fp[predict[j]] += 1



					pr = 0.0
					re = 0.0	
					for i in range(config.num_classes):

						if(tp[i] != 0):
							pr0 = float(tp[i])/(tp[i] + fp[i])
						else:
							pr0 = 0
						if(sz[i] != 0):
							re0 = float(tp[i])/sz[i]
						else:
							re0 = 0
						pr = pr + pr0
						re = re + re0

						if(tp[i] != 0 and sz[i] != 0):
							print i,namehash[i],float(tp[i])/(tp[i] + fp[i]),float(tp[i])/sz[i],sz[i]
						elif(sz[i] != 0):
							print i,namehash[i],0.0,float(tp[i])/sz[i],sz[i]

						else:
							print i,namehash[i],0.0,0.0,sz[i]

					pr = pr/config.num_classes
					re = re/config.num_classes
					print 'Macro Precision ' + str(pr)
					print 'Macro Recall ' + str(re)

					r = [hit/all_count for hit in hits]
					p_new = [hits[i]/p[i] for i in range(len(hits))] 
					for i in range(len(p_new)):
						print 'Hit ' + str(i), hits[i]
						print 'Test Precision ' + str(config.hits_k[i]), p_new[i]
						print 'Test Recall ' + str(config.hits_k[i]), r[i]
					print 'loss = ' + str(losses / num)
					#output()

if __name__ == "__main__":
	tf.app.run()