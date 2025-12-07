'''
import sqlite3
import time

import numpy as np
import pandas as pd
import tensorflow as tf
#from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoin


#gpu optimization configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Tensorflow detected {len(gpus)} GPU(s).")
else:
    print("Tensorflow did not detect any GPUs.")


current_time = str(time.time())

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../../Models/Trained-Model-OU-' + current_time, save_best_only=True, monitor='val_loss', mode='min')

dataset = "dataset_2012-24_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

OU = data['OU-Cover']
total = data['OU']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)

data['OU'] = np.asarray(total)
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(OU)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
# model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32, callbacks=[tensorboard, earlyStopping, mcp_save])

print('Done')
'''

import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

# Run the verbose detection
devices = get_available_devices()

print("--- Detected Devices ---")
for device in devices:
    print(device)
    
if any('GPU' in device for device in devices):
    print("\n✅ GPU was successfully detected!")
else:
    print("\n❌ GPU was NOT detected.")
    
# Check for low-level library detection
print("\nLow-level CUDA Check:")
print(f"Num physical GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Attempt to place a small tensor on the GPU (will fail if not detected)
try:
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        print(f"Tensor successfully created on: {a.device}")
except RuntimeError as e:
    print(f"Failed to place tensor on GPU: {e}")
