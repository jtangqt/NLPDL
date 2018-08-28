import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
from nltk import sent_tokenize
import time
import numpy as np
import pandas as pd

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
		entire_corpus = " ".join(corpus)
		handle.write(entire_corpus)

	return " ".join(corpus)


class word2vec(): 
	def __init__(self):
		self.learning_rate = settings['learning_rate']
		self.epochs = settings['i']
		self.window_size = settings['w']
		self.neg_sampling = settings['k']
		self.n = settings['n']

	def generate_data(self, settings, corpus): 
		## open up word dictionary 
			
		## cycle through each sentence in the corpus 
			## create one hot word vectors for each word in a sentence
			## create one hot word vectors for the context words
		
		i = 0
		index_to_word = {}
		word_to_index = {}
		training_data = []

		## creates the word to index dictionary and vice versa 
		for word in word_tokenize(corpus):
			if word in word_to_index: 
				continue
			else: 
				word_to_index[word] = i
				index_to_word[i] = word
				i += 1

		self.word_to_index = word_to_index
		self.index_to_word = index_to_word
		self.no_words = len(index_to_word)

		self.in_vec = np.random.uniform(-0.9, 0.9, (self.no_words, self.n))
		self.out_vec = np.random.uniform(-0.9, 0.9, (self.n, self.no_words))

		sentences = sent_tokenize(corpus)
		print(len(sentences))
		return sentences
			
	def word_one_hot(self, word): 
		vec = [0 for i in range(0, self.no_words)]
		tmp = self.word_to_index[word]
		vec[tmp] = 1
		return vec 

	def forward_prop(self, x):
		h = np.dot(self.in_vec.T, x)
		u = np.dot(self.out_vec.T, h)
		y = self.softmax(u)
		return y, u, h

	def train_data(sentences): 

		k = 0
		w_context = []
		w_target = []
		## cycle through each sentence in the corpus
		for ind_sentence in sentences: 
			sentence = word_tokenize(ind_sentence)
			len_sentence = len(sentence)
			
			for i, word in enumerate(sentence):

				# create one hot word vector for each word in a sentence 
				w_target = self.word_one_hot(word)

				#create one hot word vectors for the context words
				w_context = []
				for j in range(i - self.window_size, i + self.window_size + 1): 
					if j != i and j < len_sentence and j >= 0:
						w_context.append(self.word_one_hot(sentence[j]))

				## forward propagation 
				y, u, h = self.forward_prop(w_target)

				## calculate error

				## back propagation


		return 0


if __name__ == "__main__":
	input_file = "TC_provided"
	output_file = "corpus.txt"
	#corpus = word_to_dict(input_file, output_file)

	settings = {}

	settings['learning_rate'] = 1
	settings['i'] = 5 #epochs
	settings['w'] = 2 #window size
	settings['k'] = 10 #negative sampling
	settings['n'] = 5 #dimensionality of word embedding 

	with open(output_file, "r") as handle:
		corpus = handle.read()
	w2v = word2vec()
	token_corpus = w2v.generate_data(settings, corpus)
	w2v.train_data(token_corpus)
	print("hi")
	print(hehe)