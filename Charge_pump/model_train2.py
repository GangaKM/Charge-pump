import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)
x_train = np.loadtxt("data/input/input_data.csv",
                 delimiter=",", dtype=float)
                 
#Input and Output data
#x_train = pd.read_csv("data_new/train_input/input_data.csv", header=None)
y_train = pd.read_csv("data/output/output_data.csv", header=None)
print(x_train.shape, y_train.shape)

model = load_model("model/train4/cp_model1.h5")

callback = model.fit(x_train, y_train, epochs=100, batch_size=5)
loss = pd.DataFrame(callback.history["loss"])
loss.columns = ["loss"]
model.summary()

model.save_weights("weights/train4/train2.h5")
model.save("model/train4/cp_model2.h5")

loss.to_csv("loss_metrics/train4/loss2.csv")

