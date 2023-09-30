import pandas as pd
import os
import numpy as np

op_dir = "/home/ganga/CP_model/python_files/"
os.chdir(op_dir)

x_train = pd.read_csv("data/input/train_in10.csv", header=None)
print("Input matrix" + str(x_train.shape))
x_train = x_train.to_numpy()

w1 = pd.read_csv('weights/csv_files/weights1.csv', header=None)
w1 = w1.to_numpy()
print("Layer1 weights : "+ str(w1.shape))

b1 = pd.read_csv('weights/csv_files/bias1.csv', header=None)
b1 = b1.to_numpy()
print("Layer1 bias : " + str(b1.shape))

w2 = pd.read_csv('weights/csv_files/weights2.csv', header=None)
w2 = w2.to_numpy()
print("Layer2 weights : "+ str(w2.shape))

b2 = pd.read_csv('weights/csv_files/bias2.csv', header=None)
b2 = b2.to_numpy()
print("Layer2 bias : " + str(b2.shape))

def multiply(v, G):
    result = []
    total = 0
    for i in range(len(G)):
        total += G[i] * v[i]
    return total  

def add(v, G):
    result = np.zeros((64,1))
    for i in range(len(G)):
        result[i] = G[i] + v[i]
    return result 
    
output = np.zeros((2000))
 
for r in range(2000):
    x = x_train[r:r+1,:].T
    #x = np.ones((3000,1))
    W = np.zeros((64))

    for i in range(len(w1[0])):
         W[i] = multiply(x,w1[:,i])

    o1 = np.zeros((64,1))
    o1 = add(W,b1)
    o1[o1<0] = 0

    o2 = np.zeros((1,1))
    o2 = multiply(o1,w2)
    o2 = o2+b2
    o2[o2<0] = 0
    #print(o2)
    output[r] = o2



#i1 = x.T
#o_1 = np.zeros((64,1)).T
#o1 = np.zeros((1,64))

#for i in range(len(i1)):
#    for j in range(len(w1[0])):
#       for k in range(len(w1)):
#           o_1[i][j] += i1[i][k] * w1[k][j]
#b1 = b1.T              
#o1 = o_1 + b1
#o1[o1<0] = 0 
#o2 = np.zeros((1,1))

#for i in range(len(o1)):
#    for j in range(len(w2[0])):
#        for k in range(len(w2)):
#           o2[i][j] += o1[i][k] * w2[k][j]
#o2 = o2+b2 
#o2[o2<0] = 0
#print(o2)

out1 = pd.DataFrame(output)
out1.to_csv("predicted_outputs/matmul_results/train_out11.csv",index=False, header=False)

print(out1.shape)              
               
               
               
