import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

predicted = pd.read_csv("predicted_outputs/train4/test_in6.csv")
y_test = pd.read_csv("data/output/test_out6.csv")

#matmul = pd.read_csv("predicted_outputs/matmul_results/train_out10.csv", header=None)
#matmul1 = pd.read_csv("predicted_outputs/matmul_results/train_out11.csv", header=None)

plt.plot(y_test)
plt.plot(predicted)
plt.savefig("plots/train4/test_in6.png")
plt.show()


