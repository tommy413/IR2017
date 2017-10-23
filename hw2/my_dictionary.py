import sys
import os
import math
import my_tokenizer

corpus_path = "data/IRTM"
dictionary = set()
df = {}
wordId = {}

doc_num = len(os.listdir(corpus_path))

#make dictionary
for path in os.listdir(corpus_path):
	doc_dict = set(my_tokenizer.tokenizer(corpus_path + "/" + path))

	for w in doc_dict:
		df[w] = df.get(w,0) + 1

	dictionary = dictionary.union(doc_dict)

print("Documents Total : %d" % doc_num)
dictionary_list = sorted(list(dictionary))
print("Number of Terms : %d" % len(dictionary_list))

#dictionary.txt output
f = open("dictionary.txt", 'w')
f.write("t_index\tterm\tdf\n")
t_index = 1
for w in dictionary_list:
	f.write("%d\t%s\t%d\n" % (t_index,w,df[w]))
	wordId[w] = t_index
	t_index = t_index + 1

print("Save dictionary.txt")

#vectors output
outputPath = "vectors/"
if not (os.path.exists(outputPath)):
	os.mkdir(outputPath)

#count tf-idf and output
for path in os.listdir(corpus_path):
	docId = path.split(".")[0]
	f = open("%s%s.txt" % (outputPath,docId) , 'w')
	doc_rst = []

	word_list = my_tokenizer.tokenizer(corpus_path + "/" + path)
	l = 0.0

	tf_dict = {}
	for w in word_list:
		tf_dict[w] = tf_dict.get(w,0.0) + 1.0

	f.write("%d\n" % len(tf_dict))
	f.write("t_index\ttf-idf\n")
	for w in tf_dict.keys():
		idf = math.log(float(doc_num)/df[w] , 10 )
		l = l + idf * idf * tf_dict[w] * tf_dict[w]
		doc_rst.append([wordId[w], tf_dict[w] * idf])

	doc_rst.sort(key=lambda x: x[0])
	for row in doc_rst:
		f.write("%d\t%f\n" % (row[0],row[1]/math.sqrt(l) ) )
print("Save Vectors of Document: %d" % len(os.listdir(outputPath)))