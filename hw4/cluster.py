from my_dictionary import make_dictionary
from my_cosine import doc_vectorize
import numpy as np
import heapq
import os
import time

def pre_process(path,doc_num):
	heap = []
	token_len = make_dictionary(path,doc_num)
	vector = {}

	for i in range(0,doc_num):
		for j in range(i+1,doc_num):
			if i not in vector.keys():
				vector[i] = doc_vectorize(i+1,token_len)
			if j not in vector.keys():
				vector[j] = doc_vectorize(j+1,token_len)
			heapq.heappush(heap,(-1 * np.dot(vector[i],vector[j]), i, j) )
	return heap,vector

def count_k(rootlist):
	count = 0
	for x in range(0,len(rootlist)):
		if (x == rootlist[x][0]):
			count += 1
	return count

def find_root(rootlist,x):
	return x if rootlist[x][0] == x else find_root(rootlist,rootlist[x][0]) 

def output(rootlist,k):
	root_map = {}
	outfile = open("%d.txt" % k, 'w')

	for x in range(0,len(rootlist)):
		root = find_root(rootlist,x)
		if root not in root_map.keys():
			root_map[root] = []
		l = root_map[root]
		l.append(x+1)
		root_map[root] = l

	for c in sorted(list(root_map.keys())):
		for d in root_map[c]:
			outfile.write("%d\n" % d)
		outfile.write("\n")
	outfile.close()

def single_link(klist,heap,doc_num):
	rootlist = [[i,1] for i in range(0,doc_num)]
	count = 1

	for k in klist:
		while(count_k(rootlist) > k):
			(cs,d1,d2) = heapq.heappop(heap)
			r1 = find_root(rootlist,d1)
			r2 = find_root(rootlist,d2)
			while (r1 == r2):
				(cs,d1,d2) = heapq.heappop(heap)
				r1 = find_root(rootlist,d1)
				r2 = find_root(rootlist,d2)
			rootlist[r2] = (r1,rootlist[r2][1])
			rootlist[r1] = (r1,rootlist[r1][1]+rootlist[r2][1])
			print("Iteration %d : merge %d,%d" % (count,d1,d2))
			count += 1
		output(rootlist,k)

def centroid(klist,heap,vector,doc_num):
	rootlist = [[i,1] for i in range(0,doc_num)]
	count = 1

	for k in klist:
		while(count_k(rootlist) > k):
			#find merge pair
			(cs,d1,d2) = heapq.heappop(heap)
			r1 = find_root(rootlist,d1)
			r2 = find_root(rootlist,d2)
			while (r1 == r2):
				(cs,d1,d2) = heapq.heappop(heap)
				r1 = find_root(rootlist,d1)
				r2 = find_root(rootlist,d2)

			#merge
			rootlist[r2] = (r1,rootlist[r2][1])
			vector[r1] = (vector[r2]*rootlist[r2][1] + vector[r1]*rootlist[r1][1])/(rootlist[r2][1] + rootlist[r1][1])
			rootlist[r1] = (r1,rootlist[r1][1]+rootlist[r2][1])

			#update heap
			heap = [(cs,d1,d2) for (cs,d1,d2) in heap if d1 != r1 and d1 != r2 and d2 != r1 and d2 != r2]
			heapq.heapify(heap)
			root_points = [x for x in range(0,doc_num) if x == find_root(rootlist,x) and x != r1]
			for r in root_points:
				x1 = min(r,r1)
				x2 = max(r,r1)
				heapq.heappush(heap,(-1 * np.dot(vector[x1],vector[x2]), x1, x2) )

			print("Iteration %d : merge %d,%d" % (count,d1,d2))
			count += 1
		output(rootlist,k)

if __name__ == "__main__":
	corpus_path = "data/IRTM"
	doc_num = len(os.listdir(corpus_path))
	# doc_num = 30
	begin_time = time.time()
	heap,vector = pre_process(corpus_path,doc_num)
	print("Time comsumed : %ds" % int(time.time()-begin_time))

	Klist = [20,13,8]
	# single_link(Klist,heap,doc_num)
	centroid(Klist,heap,vector,doc_num)