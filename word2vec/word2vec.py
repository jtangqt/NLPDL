import nltk
from nltk.tokenize import word_tokenize
import os
import pickle

def word_to_dict(input_folder, output_dict):
	word_to_dict = {}
	train_list = list()
	i = 0

	os.chdir(input_folder)
	for item in os.listdir(os.getcwd()):
		if os.path.isfile(item) and item.split(".")[1] == "labels":
			train_list.append(item)

	for item in train_list:
		with open(item, 'rb') as handle:
			file_line = handle.read().splitlines()
			for file in file_line:
				file_name = file.split(" ")[0]
				train_file = open(file_name, 'rb')
				tokens = word_tokenize(train_file.read())
				train_file.close()
				for ind_tok in tokens:
					if not ind_tok in word_to_dict:
						word_to_dict[ind_tok] = i
						i += 1

	print(len(word_to_dict))

def word_to_dict_longer(input_folder, output_dict):

	mypath = input_folder
	j = 0
	for item in os.listdir(mypath):
		train_file = open(os.path.join(mypath, item), 'rb')
		tokens = word_tokenize(train_file.read())
		train_file.close()
		if j % 1000 == 0:
			print(j)

		for ind_tok in tokens:
			if not ind_tok in word_to_dict:
				word_to_dict[ind_tok] = i
				i += 1
		j += 1
		print(item)
	
	with open(output_dict, "wb") as handle:
		pickle.dump(word_to_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class word2vec(): 
	def __init__(self, settings):
		self.learning_rate = settings['learning_rate']
		self.epochs = settings['epochs']
		self.window_size = settings['window_size']
		self.corpus_name = settings['corpus_name']

	def generate_data(corpus): 
		## open up word dictionary 
			## create index to word dictionary
			## create word to index dictionary
		## cycle through each sentence in the corpus 
			## create one hot word vectors for each word in a sentence
			## create one hot word vectors for the context words


	def train_data(): 


if __name__ == "__main__":
	input_file = "TC_provided"
	output_dict = "hi"
	word_to_dict(input_file, output_dict)
