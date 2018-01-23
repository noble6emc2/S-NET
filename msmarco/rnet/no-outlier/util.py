import tensorflow as tf
import re
from collections import Counter
import string

def get_record_parser(config, is_test=False):
	def parse(example):
		para_limit = config.test_para_limit if is_test else config.para_limit
		ques_limit = config.test_ques_limit if is_test else config.ques_limit
		char_limit = config.char_limit
		features = tf.parse_single_example(example,
										   features={
											   "passage_idxs": tf.FixedLenFeature([], tf.string),
											   "ques_idxs": tf.FixedLenFeature([], tf.string),
											   "passage_char_idxs": tf.FixedLenFeature([], tf.string),
											   "ques_char_idxs": tf.FixedLenFeature([], tf.string),
											   "y1": tf.FixedLenFeature([], tf.string),
											   "y2": tf.FixedLenFeature([], tf.string),
											   "id": tf.FixedLenFeature([], tf.int64)
										   })
		passage_idxs = tf.reshape(tf.decode_raw(
			features["passage_idxs"], tf.int32), [para_limit])
		ques_idxs = tf.reshape(tf.decode_raw(
			features["ques_idxs"], tf.int32), [ques_limit])
		passage_char_idxs = tf.reshape(tf.decode_raw(
			features["passage_char_idxs"], tf.int32), [para_limit, char_limit])
		ques_char_idxs = tf.reshape(tf.decode_raw(
			features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
		y1 = tf.reshape(tf.decode_raw(
			features["y1"], tf.float32), [para_limit])
		y2 = tf.reshape(tf.decode_raw(
			features["y2"], tf.float32), [para_limit])
		qa_id = features["id"]
		return passage_idxs, ques_idxs, passage_char_idxs, ques_char_idxs, y1, y2, qa_id
	return parse


def get_batch_dataset(record_file, parser, config):
	num_threads = tf.constant(config.num_threads, dtype=tf.int32)
	dataset = tf.data.TFRecordDataset(record_file).map(
		parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
	if config.is_bucket:
		buckets = [tf.constant(num) for num in range(*config.bucket_range)]

		def key_func(passage_idxs, ques_idxs, passage_char_idxs, ques_char_idxs, y1, y2, qa_id):
			c_len = tf.reduce_sum(
				tf.cast(tf.cast(passage_idxs, tf.bool), tf.int32))
			t = tf.clip_by_value(buckets, 0, c_len)
			return tf.argmax(t)

		def reduce_func(key, elements):
			return elements.batch(config.batch_size)

		dataset = dataset.apply(tf.contrib.data.group_by_window(
			key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
	else:
		dataset = dataset.batch(config.batch_size)
	return dataset


def get_dataset(record_file, parser, config):
	num_threads = tf.constant(config.num_threads, dtype=tf.int32)
	dataset = tf.data.TFRecordDataset(record_file).map(
		parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
	return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
	answer_dict = {}
	remapped_dict = {}
	outlier = False
	for qid, p1, p2 in zip(qa_id, pp1, pp2):
		passage_concat = eval_file[str(qid)]["passage_concat"]
		spans = eval_file[str(qid)]["spans"]
		uuid = eval_file[str(qid)]["uuid"]
		spans_l = len(spans)
		if p1 >= len(spans) or p2 >= len(spans):
			continue
			#outlier = True
			p1 = p1%spans_l
			p2 = p1%spans_l
			#continue
			# it will return {},{},True
			#return answer_dict,remapped_dict,outlier
		#try:
		start_idx = spans[p1][0]
		end_idx = spans[p2][1]
		"""except:
			print(passage_concat)
			print(spans)
			print(len(spans))
			print(uuid)
			print(p1,p2)
			import sys
			sys.exit()
		"""
		answer_dict[str(qid)] = passage_concat[start_idx: end_idx]
		remapped_dict[uuid] = passage_concat[start_idx: end_idx]
	if answer_dict == {}:
		outlier = True
	return answer_dict, remapped_dict, outlier


def rouge_l(rouge_obj, prediction, ground_truth):

	prediction_tokens = normalize_answer(prediction)
	ground_truth_tokens = normalize_answer(ground_truth)
	scores = rouge_obj.get_scores(ground_truth_tokens, prediction_tokens)
	rouge_l_ = scores[0]['rouge-l']['p']
	#print(prediction_tokens)
	#print(ground_truth_tokens)
	return rouge_l_

def evaluate(eval_file, answer_dict):
	from rouge import Rouge
	
	f1 = exact_match = rouge_l_= total = 0
	rouge = Rouge()

	for key, value in answer_dict.items():
		total += 1
		ground_truths = eval_file[key]["answers"]
		prediction = value
		exact_match += metric_max_over_ground_truths(
			exact_match_score, prediction, ground_truths)
		f1 += metric_max_over_ground_truths(f1_score,
											prediction, ground_truths)
		rouge_l_ += rouge_l(rouge, prediction, ground_truths[0])
	exact_match = 100.0 * exact_match / total
	f1 = 100.0 * f1 / total
	rouge_l_ = 100.0 * rouge_l_ / total
	return {'exact_match': exact_match, 'f1': f1, 'rouge-l': rouge_l_}

def normalize_answer(s):

	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	#return white_space_fix(remove_articles(remove_punc(lower(s))))
	return lower(s)

def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def exact_match_score(prediction, ground_truth):
	return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
	scores_for_ground_truths = []
	for ground_truth in ground_truths:
		score = metric_fn(prediction, ground_truth)
		scores_for_ground_truths.append(score)
	return max(scores_for_ground_truths)
