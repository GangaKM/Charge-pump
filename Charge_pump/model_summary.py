import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

op_dir = "/home/ganga/CP_model/python_files"
os.chdir(op_dir)

model = load_model("model/train3/cp_model1.h5")

model.summary()

