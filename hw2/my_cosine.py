import math
import numpy as np
import sys

def doc_vectorize(x,y):
	xFile = open("vectors/%d.txt" % x, 'r').read()
	yFile = open("vectors/%d.txt" % y, 'r').read()

	xList = xFile.split("\n")[:-1]
	yList = yFile.split("\n")[:-1]

	l = max(int(xList[-1].split("\t")[0]),int(yList[-1].split("\t")[0]))
	print("Vector's length : %d" % l)

	vx = np.zeros(l)
	vy = np.zeros(l)

	for row in xList[2:]:
		elements = row.split("\t")
		vx[int(elements[0])-1] = float(elements[1])

	for row in yList[2:]:
		elements = row.split("\t")
		vy[int(elements[0])-1] = float(elements[1])

	return vx,vy

def cosine(x,y):
	vx,vy = doc_vectorize(x,y)
	
	return np.dot(vx,vy)

print(cosine(int(sys.argv[1]),int(sys.argv[2])))