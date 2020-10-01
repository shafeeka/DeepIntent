import os
import csv
import sys

inputf = sys.argv[1]
inputf2 = sys.argv[2]
outputf = sys.argv[3]
methods = dict()
count = 0

fin = csv.reader(open(inputf, "r"))
fin2 = open(inputf2, "r")
fout = csv.writer(open(outputf, "w"))



for line in fin2.readlines():
	line = line.strip("\n").rstrip("\t").split("\t")
	temp = list()
	for i in range(1, len(line)):
		temp.append(str(line[i]))
	methods[str(line[0])] = temp

for line in fin:
	count += 1
	print(count)
	if str(line[6]) in methods.keys():
		line[7] = methods[line[6]]
		print("yes")
		fout.writerow(line)
	else:
		print("no")
		fout.writerow(line)







	
		

