import numpy as np
import re
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

UC = []
UCu = []
UCt = []

RCr = []

EC =[]
ECu = []
ECt = []
ECl = []

wordlen = 16

print("processing char dict.....")

chars = set()

with open('test_10.txt',encoding='utf8') as f:
	for line in f.readlines():
		char = set(line)
		chars = chars | char

with open('train_10.txt',encoding='utf8') as f:
	for line in f.readlines():
		char = set(line)
		chars = chars | char

chars = sorted(list(set(chars)))
total_chars = len(chars)
print('total chars:', len(chars))
print('chars:',chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
print('char_indices',char_indices)
indices_char = dict((i, c) for i, c in enumerate(chars))
print('indices_char',indices_char)


print("processing char input ......")

with open('train_10.txt',encoding='utf8') as chfread:
	for line in chfread.readlines():
		line = line.strip().split('	')
		for i in range(1,len(line)):
			line[i] = line[i].strip().split(' ')
			linchar = []
			for word in line[i]:
				word = list(word)
				for k in range(len(word)):
					word[k] = char_indices.get(word[k])
				linchar.append(word)
			linchar = pad_sequences(linchar,maxlen = wordlen, padding = 'post')
			line[i] = linchar
		if(line[0] == '1'):
			UCt.append(line[len(line)-1])
			UCu.append(line[1:len(line)-1])
		else:
			RCr.append(line[len(line)-1])

UC.append(UCu)
UC.append(UCt)

with open('test_10.txt',encoding='utf8') as chfread:
	for line in chfread.readlines():
		line = line.strip().split('	')
		for i in range(1,len(line)):
			line[i] = line[i].strip().split(' ')
			linchar = []
			for word in line[i]:
				word = list(word)
				for k in range(len(word)):
					word[k] = char_indices.get(word[k])
				linchar.append(word)
			linchar = pad_sequences(linchar,maxlen = wordlen, padding = 'post')
			line[i] = linchar
		ECl.append(int(line[0]))
		ECu.append(line[1:len(line)-1])
		ECt.append(line[len(line)-1])

EC.append(ECu)
EC.append(ECt)
EC.append(ECl)

print("processing char_input_dict pkl dumping ...")

output_char = open('utt_char.pkl','wb')
pickle.dump(UC,output_char)
output_char.close()

output_char_re = open('re_char.pkl','wb')
pickle.dump(RCr,output_char_re)
output_char_re.close()

output_char_ev = open('ev_char.pkl','wb')
pickle.dump(EC,output_char_ev)
output_char_ev.close()

print("processing char_embedding ....")

char_embeddings_path = r"D:\PROJECT\PythonCodeProject\semantic\QANet\glove.840B.300d-char.txt"
char_embedding_dim = 300

char_embedding_vectors = {}
with open(char_embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        char_embedding_vectors[char] = vec

char_embedding_matrix = np.zeros((len(chars), 300))

for char, i in char_indices.items():
    #print ("{}, {}".format(char, i))
    char_embedding_vector = char_embedding_vectors.get(char)
    if char_embedding_vector is not None:
        char_embedding_matrix[i] = char_embedding_vector        

print("processing char_embedding pkl dumping")

char_embedding_file = open('char_embedd.pkl','wb')
pickle.dump(char_embedding_matrix,char_embedding_file)
char_embedding_file.close()