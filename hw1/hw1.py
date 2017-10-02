from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords

def input(path):
	f = open(path,'r')
	return f.read()


s = input("28.txt")
ignored = ['\'',',','.']

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
result = [t for t in stemmed if t not in stops]
# print("This is Step4's result : \n")
print(result)

fout = open("result.txt",'a');
for t in result:
	fout.write(t)
	fout.write("\n")
	
