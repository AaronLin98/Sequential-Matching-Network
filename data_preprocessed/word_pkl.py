import numpy as np
import re
import pickle

U = []
Uu = []
Ut = []

Rr = []

E = []
Eu = []
Et = []
El = []

words = set()

with open('train_10.txt',encoding='utf8') as fread:
	for line in fread.readlines():
		# print(line)
		line = line.strip().split('	')
		# print(line)
		for i in range(1,len(line)):
			line[i] = line[i].strip().split(' ')
			for word in line[i]:
				words.add(word)
		if(line[0] == '1'):
			Ut.append(line[len(line)-1])
			Uu.append(line[1:len(line)-1])
		else:
			Rr.append(line[len(line)-1])
	# U.append(Uu)
	# U.append(Ut)

# print(Uu)  # 5
# print(Ut)  # 5
print(len(words))
print("processing word add...")
# words = set()
# # Lines = []
# with open('train.txt',encoding='utf8') as fread:
# 	for line in fread.readlines():
# 		lines = line.strip().split('	')
# 		for i in lines:
# 			i = i.strip().split(' ')
# 			# Lines.append(i)
# 			for word in i:
# 			# print(word)
# 				words.add(word)

with open('test_10.txt',encoding='utf8') as fread:
	for line in fread.readlines():
		# print(line)
		line = line.strip().split('	')
		# print(line)
		for i in range(1,len(line)):
			line[i] = line[i].strip().split(' ')
			for word in line[i]:
				words.add(word)
		Eu.append(line[1:len(line)-1])
		Et.append(line[len(line)-1])
		El.append(int(line[0]))		


words = sorted(list(words))
total_words = len(words)
print('total words',total_words)
# print(words)
# print(Lines)

print("processing word dict...")

word_indices = dict((w,i) for i, w in enumerate(words))
indices_word = dict((i,w) for i, w in enumerate(words))

# print(word_indices)
# # print(indices_word)

# for i in range(len(Lines)):
# 	for j in range(len(Lines[i])):
# 		Lines[i][j] = word_indices.get(Lines[i][j])

# print(Lines)		
# print(len(Lines))
print("processing Utterance word to number...")

for i in range(len(Uu)):
	for j in range(len(Uu[i])):
		for z in range(len(Uu[i][j])):
			Uu[i][j][z] = word_indices.get(Uu[i][j][z])

# print(Uu)			
# print(len(Uu))
# print(len(Uu[0]))
# print(len(Uu[0][0]))

# print(Ut)
# print(len(Ut))

print("processing response word to number...")

for i in range(len(Ut)):
	for j in range(len(Ut[i])):
		Ut[i][j] = word_indices.get(Ut[i][j])

# print(Ut)
# print(len(Ut))		

U.append(Uu)
U.append(Ut)

print("processing nagetive word to number...")

for i in range(len(Rr)):
	for j in range(len(Rr[i])):
		Rr[i][j] = word_indices.get(Rr[i][j])

# print(Rr)
# print(len(Rr))		
print("processing evaluate word to number...")

for i in range(len(Et)):
	for j in range(len(Et[i])):
		Et[i][j] = word_indices.get(Et[i][j])

for i in range(len(Eu)):
	for j in range(len(Eu[i])):
		for z in range(len(Eu[i][j])):
			Eu[i][j][z] = word_indices.get(Eu[i][j][z])		

E.append(Eu)
E.append(Et)
E.append(El)

print("processing input_dict pkl dumping ...")

# pki file load pkl

output = open('utt.pkl','wb')
pickle.dump(U,output)
output.close()

output_re = open('re.pkl','wb')
pickle.dump(Rr,output_re)
output_re.close()

output_ev = open('ev.pkl','wb')
pickle.dump(E,output_ev)
output_ev.close()
# with open('utt.pkl', 'rb') as f:
# 	A, B = pickle.load(f)

# print(A)	
# print(B)

print("processing word_embedding...")

word_embeddings_path = r"D:\PROJECT\PythonCodeProject\semantic\QANet\glove.840B.300d.txt"
word_embedding_dim = 300

word_embedding_vectors = {}
with open(word_embeddings_path,'r',encoding='utf8') as f:
	for line in f:
		line_split = line.strip().split(" ")
		vec = np.array(line_split[1:],dtype=float)
		word = line_split[0]
		word_embedding_vectors[word] = vec

word_embedding_matrix = np.zeros((len(words),word_embedding_dim))

for word, i in word_indices.items():
	word_embedding_vector = word_embedding_vectors.get(word)
	if word_embedding_vector is not None:
		word_embedding_matrix[i] = word_embedding_vector

print("processing embedding pickle dumping....")		

embedding_file = open('embedd.pkl','wb')
pickle.dump(word_embedding_matrix,embedding_file)
embedding_file.close()