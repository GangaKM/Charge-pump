import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

train_data = pd.read_csv("data/data3.csv", header=None)

x_train = np.asarray(train_data.iloc[:,6:8])
#y_train = np.asarray(train_data.iloc[:,5])
print(x_train.shape)

input1 = np.zeros((600,1))
arr = np.zeros((600,1))

for i in range(600):
    print(i)
    arr[0] = x_train[i,0]
    arr[1] = x_train[i,1]
    pd.DataFrame(arr).T.to_csv("data/input/test_in5.csv",index=False, header=False, mode='a')
    #input1 = np.concatenate((input1,arr),axis=1)
    arr = np.roll(arr,2,axis=0)

    
#for i in range(6000):
#    print(i)
#    arr[0] = x_train[i,0]
#    arr[1] = x_train[i,1]
#    input1 = np.concatenate((input1,arr),axis=1)
#    arr = np.roll(arr,2,axis=0)
        
#input1 = input1[:,1:]
#input1 = input1.T
#print(input1.shape)

#train_in1 = pd.DataFrame(input1)
#train_in1.to_csv("data_new/test_input/test_in1.csv",index=False, header=False)

3
