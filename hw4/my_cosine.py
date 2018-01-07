import math
import numpy as np
import sys

def doc_vectorize(x,l):
	xFile = open("vectors/%d.txt" % x, 'r').read()
	
	xList = xFile.split("\n")[:-1]

	vx = np.zeros(l)

	for row in xList[2:]:
		elements = row.split("\t")
		vx[int(elements[0])-1] = float(elements[1])

	return vx

if __name__ == "__main__":
	print(np.dot(doc_vectorize(int(sys.argv[1])),doc_vectorize(int(sys.argv[2])) ) )