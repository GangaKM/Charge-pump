import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

train_data1 = pd.read_csv("data/data3.csv", header=None)
train_data = train_data1[0:600]

y_train = np.asarray(train_data.iloc[:,8])

train_out1 = pd.DataFrame(y_train)
train_out1.to_csv("data/output/test_out5.csv",index=False, header=False)

print(y_train.shape)

