from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import re
import sys

def input(path):
	f = open(path,'r')
	return f.read()

def tokenizer(path):
	s = input(path)
	ignored = [':','\\','/','*',',','.']

	for c in ignored:
		s = s.replace(c,"")
	# print("This is Step1's result : \n")
	# print(s)

	tokens = word_tokenize(s.lower())
	# print("This is Step2's result : \n")
	# print(tokens);

	porter = PorterStemmer()
	stemmed = [porter.stem(t) for t in tokens];
	# print("This is Step3's result : \n")
	# print(stemmed);

	stops = set(stopwords.words('english'))
	result = [t for t in stemmed if (t not in stops) and (re.match("^[a-zA-Z]+[-']?[a-zA-Z]?",t))]
	# print("This is Step4's result : \n")
	# print(result)

	return result
	
