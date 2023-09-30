import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

train_data = pd.read_csv("data/step2.csv", header=None)

x_train = np.asarray(train_data.iloc[:,3:5])
#y_train = np.asarray(train_data.iloc[:,5])
print(x_train.shape)

input1 = np.zeros((1300,1))
arr = np.zeros((1300,1))

for i in range(800):
    print(i)
    arr[0] = x_train[i,0]
    arr[1] = x_train[i,1]
    pd.DataFrame(arr).T.to_csv("data/input/input2.csv",index=False, header=False, mode='a')
    #input1 = np.concatenate((input1,arr),axis=1)
    arr = np.roll(arr,2,axis=0)
        


