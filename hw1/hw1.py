from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords

def input(path):
	f = open(path,'r');
	return f.read();


s = input("28.txt");
ignored = ['\'',',','.'];

for c in ignored:
	s = s.replace(c,"");


tokens = word_tokenize(s.lower());
porter = PorterStemmer();
stemmed = [porter.stem(t) for t in tokens];
print(stemmed);

stops = set(stopwords.words('english'))
print([t for t in stemmed if t not in stops])

