import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

model = load_model("model/train4/cp_model1.h5")
model.summary()

weights1 = pd.DataFrame(model.layers[0].get_weights()[0])
bias1 = pd.DataFrame(model.layers[0].get_weights()[1])
print(weights1.shape,bias1.shape)
weights1.to_csv('weights1.csv', index=False, header=False)
bias1.to_csv('bias1.csv', index=False, header=False)

weights2 = pd.DataFrame(model.layers[1].get_weights()[0])
bias2 = pd.DataFrame(model.layers[1].get_weights()[1])
print(weights2.shape,bias2.shape)
weights2.to_csv('weights2.csv', index=False, header=False)
bias2.to_csv('bias2.csv', index=False, header=False)
