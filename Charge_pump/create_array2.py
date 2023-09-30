# seperate data as seprate columns/rows

import csv
import numpy as np
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

with open('weights1.csv',"r") as csvfile:
	reader = csv.reader(csvfile)
	header = next(reader)
	columns = [[] for _ in range(len(header))]
	for row in reader:
		for i in range(len(row)):
			columns[i].append(row[i])		


#for i, col in enumerate(columns):
#	print(f"Column {i+1}: {col}")

#print(columns[1])
# convert to VerilogA 	
for i in range(16):
	data = np.array(columns[i]) # accepts single parameter 
	#print(data.size)	
	with open("weights1.v", 'a') as f:
		f.write("w1[0:600] = '{")
		f.write(', '.join([str(float(x)) for x in data]))
		f.write("};\n")
#print(f)
