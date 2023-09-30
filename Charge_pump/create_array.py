# seperate data as seprate columns/rows

import csv
import numpy as np
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

with open('weights1.csv',"r") as csvfile:
	reader =  csv.reader(csvfile)
	header = next(reader)
	columns = [[] for _ in header]
	with open("bias_files.txt",'a') as g:
		for row in reader:
			for i, value in enumerate(row):
				columns[i].append(value)
		g.write(str(columns[i]))
		


#for i, col in enumerate(columns):
#	print(f"Column {i}: {col}")

#print(columns[1])
# convert to VerilogA 	
for i in range(len(columns)):
	data = np.array(columns[i]) # accepts single parameter 
	#print(data.size)	
	with open("weights1.v", 'a') as f:
		f.write("data[16:0] = '{")
		f.write(', '.join([str(float(x)) for x in data]))
		f.write("};\n")
#print(f)
