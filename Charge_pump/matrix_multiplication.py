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
b1 = b1.T
output = np.zeros((2000))
for r in range(2000):
    i1 = x_train[r:r+1,:]
    #i1.reshape(1,3000)
    #print("Input shape : " + str(i1.shape))

    # Using in-built functions
    #o3 = np.dot(i1, w1) + b1.T         
    #o3[o3<0] = 0
    #print(o3)
    #o4 = np.dot(o3, w2) + b2
    #print(o4)

    # Without in-built functions
    o_1 = np.zeros((64,1)).T
    o1 = np.zeros((1,64))

    # Layer 1 output
    # input and weight matrix mutliplication
    for i in range(len(i1)):
        for j in range(len(w1[0])):
           for k in range(len(w1)):
               o_1[i][j] += i1[i][k] * w1[k][j]
    
    #print("Adding bias.....")
    # adding bias
    #for i in range(len(o_1)):
     #  for j in range(len(o_1[0])):
      #     o1[i][j] = o_1[i][j] + b1[i][j]
    o1 = o_1 + b1
       
    #print("Layer1 output : " + str(o1.shape))
    o1[o1<0] = 0 # ReLU 
    #print(o1)

    #Layer 2 output
    o2 = np.zeros((1,1))

    for i in range(len(o1)):
        for j in range(len(w2[0])):
            for k in range(len(w2)):
               o2[i][j] += o1[i][k] * w2[k][j]
    o2 = o2+b2 # adding bias
    o2[o2<0] = 0
    #print("Layer2 output : " +str(o2.shape))
    #print("Value : " + str(o2))
    output[r] = o2
    #print(o2)
    
print(output)
out1 = pd.DataFrame(output)
out1.to_csv("predicted_outputs/matmul_results/train_out10.csv",index=False, header=False)

print(out1.shape)




