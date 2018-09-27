import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
from nltk import sent_tokenize
import time
import numpy as np
import pandas as pd
import math

def word_to_dict_longer(input_folder, output_dict):

	mypath = input_folder
	j = 0
	corpus = {}

	for item in os.listdir(mypath):
		train_file = open(os.path.join(mypath, item), 'rb')
		tokens = word_tokenize(train_file.read())
		corpus.append(tokens)
		train_file.close()
		if j % 1000 == 0:
			print(j)

		for ind_tok in tokens:
			ind_tok = ind_tok.lower()
			if not ind_tok in word_to_dict:
				word_to_dict[ind_tok] = i
				i += 1
		j += 1
		print(item)
	
	with open(output_dict, "wb") as handle:
		pickle.dump(word_to_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return " ".join(corpus)


def word_to_dict(input_folder, output_file):
	word_to_dict = {}
	train_list = list()
	corpus = list()
	i = 0

	os.chdir(input_folder)

	for item in os.listdir(os.getcwd()):
		if os.path.isfile(item) and item.split(".")[1] == "labels":
			train_list.append(item)

	for item in train_list:
		with open(item, 'r') as handle:
			file_line = handle.read().splitlines()
			for file in file_line:
				file_name = file.split(" ")[0]
				
				train_file = open(file_name, 'r')
				data = train_file.read()
				corpus += data.split(" ")
				train_file.close()

	os.chdir("../")

	with open(output_file, 'w+') as handle:
		entire_corpus = " ".join(corpus).lower()
		handle.write(entire_corpus)

class vocab:
	def __init__(self, word):
		self.word = word
		self.count = 0
	def add_count(self):
		self.count += 1

class TableForNegativeSamples:
	def __init__(self, vocab, index_to_word, vocab_dict):
		power = 0.75
		norm = sum([math.pow(vocab_dict[index_to_word[t]].count, power) for t in vocab]) # Normalizing constants

		table_size = 10000000
		table = np.zeros(table_size, dtype=np.uint32)

		p = 0 # Cumulative probability
		i = 0
		for j, word in enumerate(vocab):
			p += float(math.pow(float(vocab_dict[index_to_word[word]].count), power))/norm
			while i < table_size and float(i) / table_size < p:
				table[i] = j
				i += 1
		self.table = table

	def sample(self, count, length):
		# create all of the random vectors
		indices = np.random.randint(low=0, high=len(self.table), size=(length, count))
		return [self.table[i] for i in indices]


class word2vec(): 
	def __init__(self):
		self.learning_rate = settings['learning_rate']
		self.epochs = settings['i']
		self.window_size = settings['w']
		self.neg_sampling = settings['k']
		self.n = settings['n']

	def generate_data(self, settings, corpus): 
		i = 0
		index_to_word = {}
		word_to_index = {}
		training_data = []
		vocab_dict = {}
		tmp_vocab_dict = [None for x in range(len(corpus))]

		## creates the word to index dictionary and vice versa 
		for word in word_tokenize(corpus):
			if word in vocab_dict: 
				vocab_dict[word].add_count()
			else: 
				tmp_vocab_dict[i] = vocab(word) 
				vocab_dict[word] = tmp_vocab_dict[i]
				word_to_index[word] = i
				index_to_word[i] = word
				i += 1

		self.vocab_dict = vocab_dict
		self.word_to_index = word_to_index
		self.index_to_word = index_to_word
		self.no_words = len(index_to_word)

		self.in_vec = np.random.uniform(-0.9/100, 0.9/100, (self.no_words, self.n))
		self.out_vec = np.random.uniform(-0.9/100, 0.9/100, (self.no_words, self.n))

	def word_one_hot(self, word): 
		vec = [0 for i in range(0, self.no_words)]
		tmp = self.word_to_index[word]
		vec[tmp] = 1
		return vec 

	def softmax(self, x):
		num = np.exp(x)
		return num / num.sum(axis = 0)

	def forward_prop(self, x):
		v_c = np.dot(self.in_vec, x)
		u_v = np.dot(self.out_vec.T, v_c)
		y = self.softmax(u_v)
		return y, u_v, v_c

	def sigmoid(self, x):
		return 1 / (np.exp(-x) + 1)

	def train_data(self, corpus): 

		epoch = 0
		w_context = []
		w_target = []
		
		# Initialize variables 
		len_tokens = len(word_tokenize(corpus))
		tokens = [word[1] for word in self.word_to_index.items()]
		sentences = sent_tokenize(corpus)
		table = TableForNegativeSamples(tokens, self.index_to_word, self.vocab_dict)

		while epoch < self.epochs: 
			
			neg_sampling = table.sample(settings['k'], len_tokens * (2*self.window_size+1))
			word_count = 0
			loss = 0

			# Loop through each sentence
			for ind_sentence in sentences: 
				sentence = word_tokenize(ind_sentence) # [:-1]
				len_sentence = len(sentence)
				
				# Loop through each word in the sentence as target
				for i, word in enumerate(sentence):
					k = 0

					# Loop through each context word for the target 
					for j in range(i - self.window_size, i + self.window_size + 1): 
						if j != i and j < len_sentence and j >= 0:
							context = sentence[j]
							classifiers = [(self.word_to_index[word], 1)] + [(target, 0) for target in neg_sampling[word_count * self.window_size + k]]
							neu1e = np.zeros(self.n)
							q_i = 0

							for target, label in classifiers:
								nn0 = self.in_vec[self.word_to_index[context]]
								nn1 = self.out_vec[target]
								z = np.dot(nn0, nn1)
								p = self.sigmoid(z)
								g = self.learning_rate * (label - p)

								if label == 1:
									p_i = p
								else:
									q_i += np.log(1-p)

								neu1e += g * nn1 
								self.out_vec[target] += g * nn0

							self.in_vec[self.word_to_index[context]] += neu1e
							loss += -np.log(p_i) - q_i

							k += 1
					word_count += 1
			print("finished {0} epoch, loss: {1}".format(epoch, loss))
			epoch += 1

		pass

    # input word, returns top [n] most similar words
	def word_sim(self, word, top_n):
		w1_index = self.word_to_index[word]
		v_w1 = self.in_vec[w1_index]

		# CYCLE THROUGH VOCAB
		word_sim = {}
		for i in range(self.no_words):
			v_w2 = self.out_vec[i]
			theta_num = np.dot(v_w1, v_w2)
			theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
			theta = theta_num / theta_den

			tmp_word = self.index_to_word[i]
			word_sim[tmp_word] = theta

		words_sorted = sorted(word_sim.items(), key=lambda word : word[1], reverse=True)

		i = 0
		for item, sim in words_sorted:
			print(item, sim)
			i += 1
			if i > top_n:
				break

		pass

	def word_vec(self, word):
		w_index = self.word_to_index[word]
		v_w = self.in_vec[w_index]
		return v_w

	def vec_word(self, vec, k):
		word_sim = {}
		for i in range(self.no_words):
			v_w2 = self.in_vec[i]
			theta_num = np.dot(vec, v_w2)
			theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
			theta = theta_num/theta_den
			word = self.index_to_word[i]
			word_sim[word] = theta

		words_sorted = sorted(word_sim.items(), key=lambda word : word[1], reverse=True)

		return words_sorted[:k]

	def word_dist(self, word_a, word_b):
		a = self.in_vec[self.word_to_index[word_a]]
		b = self.in_vec[self.word_to_index[word_b]]
		num = np.dot(a,b)
		denom = np.linalg.norm(a) * np.linalg.norm(b)
		return num/denom

	def analogies(self, word_a, word_b, word_d, top_n):
		a = self.in_vec[self.word_to_index[word_a]]
		b = self.in_vec[self.word_to_index[word_b]]
		d = self.in_vec[self.word_to_index[word_d]]
		c = a - b + d 

		i = 0
		for item, sim in self.vec_word(c, top_n):
			if item == ".": 
				continue
			print(item, sim)
			i += 1
			if i > top_n:
				break

		return self.vec_word(c, top_n) #return word c

def save(w2v, output_pkl):
	with open(output_pkl, 'wb') as handle:
		pickle.dump(w2v, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	input_file = "TC_provided"
	output_file = "corpus1.txt"
	# corpus = word_to_dict(input_file, output_file)

	settings = {}

	settings['learning_rate'] = 0.05
	settings['i'] = 20 #epochs
	settings['w'] = 3 #window size
	settings['k'] = 2 #negative sampling
	settings['n'] = 7 #dimensionality of word embedding 

	# with open(output_file, "r") as handle:
	# 	corpus = handle.read()

	# # corpus = "the quick brown fox jumped over the lazy dog."
	# w2v = word2vec()
	# w2v.generate_data(settings, corpus)
	# w2v.train_data(corpus)
	# save(w2v, "0.05-7-2.pkl")

	with open("0.05-5-2.pkl", "rb") as handle:
		w2v = pickle.load(handle)

	word = "dog"
	print(word)
	w2v.word_sim(word, 30)

	# a = "sky"
	# b = "blue"
	# d = "red"
	# print(a, b, d)
	# w2v.analogies(a, b, d, 40)

	# a_1 = "pasta"
	# a_2 = "car"
	# print("{0}, {1}".format(a_1, a_2))
	# print(w2v.word_dist(a_1, a_2))

	# b_1 = "rainbow"
	# b_2 = "see"
	# print("{0}, {1}".format(b_1, b_2))
	# print(w2v.word_dist(b_1, b_2))