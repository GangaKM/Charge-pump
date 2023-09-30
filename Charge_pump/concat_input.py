import pandas as pd
import os

op_dir = "/home/ganga/python_files/codes/data/input"
os.chdir(op_dir)
op_file_name="input_data.csv"

data1 = pd.read_csv("train_in1.csv", header=None)
data2 = pd.read_csv("train_in2.csv", header=None)
data3 = pd.read_csv("train_in3.csv", header=None)
data4 = pd.read_csv("train_in4.csv", header=None)
data5 = pd.read_csv("train_in5.csv", header=None)
data6 = pd.read_csv("train_in6.csv", header=None)
data7 = pd.read_csv("train_in7.csv", header=None)
data8 = pd.read_csv("train_in8.csv", header=None)
data9 = pd.read_csv("train_in9.csv", header=None)
data10 = pd.read_csv("train_in10.csv", header=None)
data11 = pd.read_csv("train_in11.csv", header=None)
data12 = pd.read_csv("train_in12.csv", header=None)
data13 = pd.read_csv("train_in13.csv", header=None)
data14 = pd.read_csv("train_in14.csv", header=None)
data15 = pd.read_csv("train_in15.csv", header=None)
data16 = pd.read_csv("train_in16.csv", header=None)
data17 = pd.read_csv("train_in17.csv", header=None)
data18 = pd.read_csv("train_in18.csv", header=None)
#data19 = pd.read_csv("train_in19.csv", header=None)
#data20 = pd.read_csv("train_in20.csv", header=None)
#data21 = pd.read_csv("train_in21.csv", header=None)

input_data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18], axis=0)
#data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21
#print(data13.shape, data14.shape, data15.shape, data16.shape, data17.shape)
print(input_data.shape)
print(data1.shape, data2.shape, data3.shape, data4.shape, data5.shape, data6.shape, data7.shape, data8.shape, data9.shape, data10.shape, data11.shape, data12.shape, data13.shape, data14.shape, data15.shape, data16.shape, data17.shape, data18.shape)

input_data.to_csv(op_file_name, index=False, header=False)
