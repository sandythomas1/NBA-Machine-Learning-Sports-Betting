import sqlite3
import time

import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# gpu optimization configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Enable memory growth (avoids allocating all memory at once)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"**Successfully configured TensorFlow to use GPU: {gpus[0].name}**")
    except RuntimeError as e:
        print(e)

current_time = str(time.time())

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# FIX: Append .keras extension to the file path
mcp_save = ModelCheckpoint('../../Models/Trained-Model-ML-' + current_time + '.keras', save_best_only=True, monitor='val_loss', mode='min') 

dataset = "dataset_2012-24_new"
con = sqlite3.connect("/mnt/c/Users/sandy/Desktop/dev/NBA-Machine-Learning-Sports-Betting/Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

scores = data['Score']
margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU', 'OU-Cover'], axis=1, inplace=True)

data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(margin)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# batch_size is correctly set to 256 for GPU training
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=1024, callbacks=[tensorboard, earlyStopping, mcp_save]) 

print('Done')