import numpy as np
import os
import my_tokenizer
import math
import random

nb_class = 13
nb_feature = 500
nb_doc = 1095
train_size = 12

def split_set(path):
	train_dict, valid_dict, test_dict = dict(),dict(),dict()

	train_str = open(path,'r').read()
	train_c = [ [int(i) for i in line.split(" ") if i != ''] for line in train_str.split("\n")]

	for c in train_c:
		class_id = c[0]
		shuffle = c[1:]
		random.shuffle(shuffle)
		for i in shuffle[:train_size]:
			train_dict[i] = class_id
		if train_size < 15:
			for i in shuffle[train_size:]:
				valid_dict[i] = class_id

	for i in range(1,nb_doc+1):
		if i not in list(train_dict.keys()) and i not in list(valid_dict.keys()):
			test_dict[i] = 0
	
	return train_dict, valid_dict, test_dict
	# return dict (id : c)
	# class_id 0 for unknown(test)


def make_dictionary(id_list):
	# id_dict (id : (token : freq))
	id_dict = dict()

	for i in id_list:
		id_dict[i] = dict()
		data_path = os.path.join("IRTM","%d.txt" % i)
		id_tokens = my_tokenizer.tokenizer(data_path)

		for t in id_tokens:
			id_dict[i][t] = id_dict[i].get(t,0) + 1

	# dictionary (token : (id : freq))
	dictionary = dict()
	for i in range(0,len(id_list)):
		for t in dictionary.keys():
			dictionary[t][id_list[i]] = 0

		for t in id_dict[id_list[i]].keys():
			# init (id : freq)
			if t not in dictionary.keys():
				dictionary[t] = dict()
				for j in range(0,i+1):
					dictionary[t][id_list[j]] = 0
			dictionary[t][id_list[i]] = dictionary[t].get(id_list[i],0) + 1

	return dictionary

def likelihood_selection(train_dict, dictionary):
	t_list = list(dictionary.keys())
	c_list = [i for i in range(1,nb_class+1)]

	lh_dict = dict()
	for t in t_list:
		lh_list = []
		for c in c_list:
			# count n*4
			n = [[0.0,0.0],[0.0,0.0]]
			for ix in train_dict.keys():
				c_cond = train_dict[ix] == c
				t_cond = dictionary[t][ix] > 0
				n[1 if c_cond else 0][1 if t_cond else 0] += 1
			
			# pt,p1,p2
			n_sum = len(train_dict.keys())
			pt = (n[1][1] + n[0][1]) / n_sum
			p1 = n[1][1] / (n[1][1] + n[1][0])
			p2 = n[0][1] / (n[0][1] + n[0][0])

			# cal LH
			lh = -2.0 * math.log( math.pow(pt,n[1][1]) * math.pow((1.0-pt),n[1][0]) * math.pow(pt,n[0][1]) * math.pow((1.0-pt),n[0][0]) / math.pow(p1,n[1][1]) * math.pow((1.0-p1),n[1][0]) * math.pow(p2,n[0][1]) * math.pow((1.0-p2),n[0][0]) )
			lh_list.append(lh)
		lh_list = np.array(lh_list)
		lh_dict[t] = lh_list.sum()
	
	lh_rst = [[i[0],i[1]] for i in lh_dict.items()]
	lh_rst.sort(key=lambda x: x[1],reverse=True)
	feature = [p[0] for p in lh_rst[:nb_feature]]

	return feature

def train(train_dict,tokens):
	# make class_dict (c : (t : n)) (has no key t if t does not appear)
	class_dict = dict()

	for i in train_dict.keys():
		if not train_dict[i] in class_dict.keys():
			class_dict[train_dict[i]] = dict()

		data_path = os.path.join("IRTM","%d.txt" % i)
		id_tokens = [t for t in my_tokenizer.tokenizer(data_path) if t in tokens]

		for t in id_tokens:
			class_dict[train_dict[i]][t] = class_dict[train_dict[i]].get(t, 0) + 1
	
	# make model (t : (c : p))
	model = dict()
	for t in tokens:
		model[t] = dict()

	for c in class_dict.keys():
		c_sum = 0
		for c_t in class_dict[c].keys():
			c_sum = c_sum + class_dict[c][c_t]

		for t in tokens:
			model[t][c] = float(class_dict[c].get(t,0) + 1.0)/float(c_sum + nb_feature)
		

	return model

def build_model(train_dict):
	# make dictionary (tokens : (id : freq)) 
	dictionary = make_dictionary(list(train_dict.keys()))
	print("Number of Terms in Training Dataset - %d" % len(dictionary.keys()))

	# return tokens after selecting
	tokens = likelihood_selection(train_dict, dictionary)
	print("Terms after Feature Selection :", tokens )

	# return model (t : (c : p))
	model = train(train_dict,tokens)

	return model


def predict(model,doc_dict):
	feature = list(model.keys())
	rst_dict = dict()

	for i in doc_dict.keys():
		data_path = os.path.join("IRTM","%d.txt" % i)
		id_tokens = [t for t in my_tokenizer.tokenizer(data_path) if t in feature]

		rst = []
		for c in range(1,nb_class+1):
			p = 0
			for t in id_tokens:
				p = p + math.log(model[t][c])
			rst.append(p)
		#print(i,len(id_tokens),rst)
		rst = np.array(rst)
		rst_dict[i] = np.argmax(rst) + 1

	return rst_dict


def evaluation(valid_dict,val_result):
	n = len(valid_dict.keys())
	corr = 0

	for i in valid_dict.keys():
		if valid_dict[i] == val_result[i] :
			corr += 1

	print("Accuracy : %f" % (float(corr*100)/n) )


def output(test_result):
	fout = open("B03705012.txt",'w')
	id_list = sorted(list(test_result.keys()))
	#fout.write("doc_id\tclass_id\n")

	for i in id_list:
		fout.write("%d\t%d\n" % (i,test_result[i]))

	print("Prediction of Testing Dataset Finished.")
	fout.close()


def main():
	label_path = "training.txt"

	train_dict, valid_dict, test_dict = split_set(label_path)
	print("Number of Documents :: Training Dataset - %d | Validation - %d | Testing Dataset - %d" % (len(train_dict.keys() ), len(valid_dict.keys() ), len(test_dict.keys() ) ) )
	model = build_model(train_dict)
	if train_size < 15 :
		val_result = predict(model,valid_dict)
		evaluation(valid_dict,val_result)
	test_result = predict(model,test_dict)
	output(test_result)

main()