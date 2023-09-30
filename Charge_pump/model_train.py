import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from keras import regularizers

op_dir = "/home/ganga/python_files/codes"
os.chdir(op_dir)

import numpy as np
 
 
# using loadtxt()
x_train = np.loadtxt("data/input/input_data.csv",
                 delimiter=",", dtype=float)
                 
#Input and Output data
#x_train = pd.read_csv("data_new/train_input/input_data.csv", header=None)
y_train = pd.read_csv("data/output/output_data.csv", header=None)
print(x_train.shape, y_train.shape)


#Define model
model = tf.keras.models.Sequential() # Sequential = feedforward, no backward dataflow
model.add(tf.keras.layers.Dense(units=16, activation = tf.nn.relu)) # Input layer
#model.add(tf.keras.layers.Dense(units=64, activation = tf.nn.relu)) # Hidden layer
model.add(tf.keras.layers.Dense(units = 1, activation=tf.nn.relu)) # Output layer

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt ,loss = 'mse')

callback = model.fit(x_train, y_train, epochs=5, batch_size=5)
loss = pd.DataFrame(callback.history["loss"])
loss.columns = ["loss"]
model.summary()


model.save_weights("weights/train4/train1.h5")
model.save("model/train4/cp_model1.h5")

loss.to_csv("loss_metrics/train4/loss1.csv")

