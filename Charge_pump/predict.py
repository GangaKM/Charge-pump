import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

x_train = pd.read_csv("data/input/test_in8.csv", header=None)

model = load_model("model/train4/cp_model1.h5")
model.summary()

pred = model.predict(x_train)
print(pred.shape)
  

#np.savetxt("predicted_outputs/predict_train_in10.csv", pred, delimiter = ",") 
np.savetxt("predicted_outputs/train4/test_in8.csv", pred, delimiter = ",") 
