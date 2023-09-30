import pandas as pd
import os

op_dir = "/home/ganga/python_files/codes/data/output/"
os.chdir(op_dir)
op_file_name="output_data.csv"

data1 = pd.read_csv("train_out1.csv", header=None)
data2 = pd.read_csv("train_out2.csv", header=None)
data3 = pd.read_csv("train_out3.csv", header=None)
data4 = pd.read_csv("train_out4.csv", header=None)
data5 = pd.read_csv("train_out5.csv", header=None)
data6 = pd.read_csv("train_out6.csv", header=None)
data7 = pd.read_csv("train_out7.csv", header=None)
data8 = pd.read_csv("train_out8.csv", header=None)
data9 = pd.read_csv("train_out9.csv", header=None)
data10 = pd.read_csv("train_out10.csv", header=None)
data11 = pd.read_csv("train_out11.csv", header=None)
data12 = pd.read_csv("train_out12.csv", header=None)
data13 = pd.read_csv("train_out13.csv", header=None)
data14 = pd.read_csv("train_out14.csv", header=None)
data15 = pd.read_csv("train_out15.csv", header=None)
data16 = pd.read_csv("train_out16.csv", header=None)
data17 = pd.read_csv("train_out17.csv", header=None)
data18 = pd.read_csv("train_out18.csv", header=None)
#data19 = pd.read_csv("train_out19.csv", header=None)
#data20 = pd.read_csv("train_out20.csv", header=None)
#data21 = pd.read_csv("train_out21.csv", header=None)

output_data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18], axis=0)
#data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21

print(data1.shape, data2.shape, data3.shape, data4.shape, data5.shape, data6.shape, data7.shape, data8.shape, data9.shape, data10.shape, data11.shape, data12.shape, data13.shape, data14.shape, data15.shape, data16.shape, data17.shape, data18.shape)

#data1.shape, data2.shape, data3.shape, data4.shape, data5.shape, data6.shape, data7.shape, data8.shape, data9.shape, data10.shape, data11.shape, data12.shape, data13.shape, data14.shape, data15.shape, data16.shape, data17.shape, data18.shape, data19.shape, data20.shape, data21.shape
print(output_data.shape)

output_data.to_csv(op_file_name, index=False, header=False)
