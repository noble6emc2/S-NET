import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, attention, dense


class Model(object):
	def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
		self.config = config
		self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
										   initializer=tf.constant_initializer(0), trainable=False)
		self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id, \
			self.c_pr, self.ch_pr, self.pr, self.y1_pr, self.y2_pr = batch.get_next()
		self.is_train = tf.get_variable(
			"is_train", shape=[], dtype=tf.bool, trainable=False)
		self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
			word_mat, dtype=tf.float32), trainable=False)
		self.char_mat = tf.get_variable(
			"char_mat", char_mat.shape, dtype=tf.float32)

		self.c_mask = tf.cast(self.c, tf.bool)
		self.q_mask = tf.cast(self.q, tf.bool)
		
		# passage ranking line:
		#self.pr_mask = tf.cast(self.p, tf.bool)

		self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
		self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

		if opt:
			N, CL = config.batch_size, config.char_limit
			self.c_maxlen = tf.reduce_max(self.c_len)
			self.q_maxlen = tf.reduce_max(self.q_len)
			self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
			self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
			self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
			self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
			self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
			self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
			self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
			self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])

			# passage ranking
			#print(self.ch_pr.get_shape())
			#print(self.c_pr.get_shape())
			self.c_pr = tf.slice(self.c_pr, [0, 0], [N, config.max_para*config.para_limit])
			self.ch_pr = tf.slice(self.ch_pr, [0, 0, 0], [N, config.max_para*config.para_limit, CL])
		else:
			self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

		self.ch_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
		self.qh_len = tf.reshape(tf.reduce_sum(
			tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

		self.ready()

		if trainable:
			self.lr = tf.get_variable(
				"lr", shape=[], dtype=tf.float32, trainable=False)
			self.opt = tf.train.AdadeltaOptimizer(
				learning_rate=self.lr, epsilon=1e-6)
			grads = self.opt.compute_gradients(self.loss)
			gradients, variables = zip(*grads)
			capped_grads, _ = tf.clip_by_global_norm(
				gradients, config.grad_clip)
			self.train_op = self.opt.apply_gradients(
				zip(capped_grads, variables), global_step=self.global_step)

	def ready(self):
		config = self.config
		N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
		gru = cudnn_gru if config.use_cudnn else native_gru

		gi = []
		att_vP = []
		
		for i in range(config.max_para):
			print(i)
			with tf.variable_scope("emb"+str(i)):
				with tf.variable_scope("char"+str(i)):
					#CL = tf.Print(CL,[CL],message="CL:")
					#PL = tf.Print(PL,[PL],message="PL:")
					#self.ch_pr = tf.Print(self.ch_pr,[self.ch_pr.get_shape()],message="ch_pr:")
					self.ch_pr = self.ch_pr[:,:(i+1)*400,:]
					print(self.ch_pr[:,:(i+1)*400,:].get_shape())
					#self.c_pr = tf.reshape(self.c_pr, [N, 12, PL])
					#print(self.ch.get_shape())
					#print(self.ch_pr.get_shape())
					#print(self.c.get_shape())
					#print(self.c_pr.get_shape())
					ch_emb = tf.reshape(tf.nn.embedding_lookup(\
						self.char_mat, self.ch_pr), [N * PL, CL, dc])
					#	self.char_mat, self.ch), [N * PL, CL, dc])
					qh_emb = tf.reshape(tf.nn.embedding_lookup(
						self.char_mat, self.qh), [N * QL, CL, dc])
					ch_emb = dropout(
						ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
					qh_emb = dropout(
						qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
					cell_fw = tf.contrib.rnn.GRUCell(dg)
					cell_bw = tf.contrib.rnn.GRUCell(dg)
					_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
						cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
					ch_emb = tf.concat([state_fw, state_bw], axis=1)
					_, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
						cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
					qh_emb = tf.concat([state_fw, state_bw], axis=1)
					qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
					ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

				with tf.name_scope("word"+str(i)):
					c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
					q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

				c_emb = tf.concat([c_emb, ch_emb], axis=2)
				q_emb = tf.concat([q_emb, qh_emb], axis=2)

			with tf.variable_scope("encoding"+str(i)):
				rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				c = rnn(c_emb, seq_len=self.c_len)
				q = rnn(q_emb, seq_len=self.q_len)

			with tf.variable_scope("attention"+str(i)):
				qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
									   keep_prob=config.keep_prob, is_train=self.is_train)
				rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				att = rnn(qc_att, seq_len=self.c_len)
				# att is the v_P
				if i==0:
					att_vP = att
				else:
					att_vP = tf.concat([att_vP, att], axis=2)

			"""
			with tf.variable_scope("match"):
				self_att = dot_attention(
					att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
				rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
				).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
				match = rnn(self_att, seq_len=self.c_len)
			"""
		with tf.variable_scope("pointer"):

			# r_Q:
			init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
						keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			
			pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
			)[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
			logits1, logits2 = pointer(init, att, d, self.c_mask)

		with tf.variable_scope("predict"):
			outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
							  tf.expand_dims(tf.nn.softmax(logits2), axis=1))
			outer = tf.matrix_band_part(outer, 0, 15)
			self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
			self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
			losses = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits1, labels=self.y1)
			losses2 = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits2, labels=self.y2)
			self.loss = tf.reduce_mean(losses + losses2)

			# print losses
			#condition = tf.greater(self.loss, 11)
			#self.yp1 = tf.where(condition, tf.Print(self.yp1,[self.yp1],message="Yp1:"), self.yp1)
			#self.yp2 = tf.where(condition, tf.Print(self.yp2,[self.yp2],message="Yp2:"), self.yp1)
		
		if config.with_passage_ranking:
			for i in range(config.max_para):
				# Passage ranking
				with tf.variable_scope("passage-ranking-attention"+str(i)):
					vj_P = dropout(att_vP[i], keep_prob=keep_prob, is_train=is_train)
					r_Q = dropout(init, keep_prob=keep_prob, is_train=is_train)
					r_P = attention(r_Q, vj_P, mask=self.c_mask, hidden=d,
						keep_prob=config.keep_prob, is_train=self.is_train)

					#rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=pr_att.get_shape(
					#).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
					#att_rp = rnn(qc_att, seq_len=self.c_len)

					# Wg
					concatenate = tf.concat([init,att_rp],axis=2)
					g = tf.nn.tanh(dense(concatenate, hidden=d, use_bias=False, scope="g"+str(i)))
					g_ = dense(g, 1, use_bias=False, scope="g_"+str(i))
					gi.append(g_)
			gi_ = tf.convert_to_tensor(gi)
			gi = tf.nn.softmax(gi_)
			self.pr_loss = tf.nn.softmax_cross_entropy_with_logits(
						logits=gi, labels=self.pr)

	def print(self):
		pass

	def get_loss(self):
		return self.loss

	def get_global_step(self):
		return self.global_step
