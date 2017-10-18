import sys
import os
import my_tokenizer

corpus_path = "data/IRTM"
dictionary = set()

for path in os.listdir(corpus_path):
	dictionary = dictionary.union(my_tokenizer.tokenizer(corpus_path + "/" + path))

print("result:")
print(sorted(list(dictionary)))