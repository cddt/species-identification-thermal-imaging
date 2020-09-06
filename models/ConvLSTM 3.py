from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime
from matplotlib import pyplot as plt

"""
Predicts the class using ConvLSTM.
https://thebinarynotes.com/video-classification-keras-convlstm/
"""

def load(name):
    # Loads the videos and converts the labels into one-hot encoding for Keras
    X = np.load("/home/cddt/data-space/COMPSCI760/cacophony-preprocessed/" + name + ".npy")
    y = np.load("/home/cddt/data-space/COMPSCI760/cacophony-preprocessed/" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X, y_one_hot_encoded

epochs = 30
batch_size = 32
learning_rate = 0.001

# Loads the preprocessed datasets
print("Dataset loading..", end = " ")
X_train, y_train = load("training")
X_val, y_val = load("validation")
X_test, y_test = load("test")
print("Dataset loaded!")

model = Sequential()
model.add(ConvLSTM2D(32, (3,3), data_format = "channels_first", input_shape = (45,3,24,24), return_sequences = True))
model.add(Dropout(0.5))
model.add(ConvLSTM2D(64, (3,3), data_format = "channels_first", input_shape = (45,3,24,24), return_sequences = False))
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(256, activation="relu"))
#model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])

print(model.summary())

# create log dir
if not os.path.exists("./logs"):
    os.makedirs("./logs")

current_time = str(datetime.datetime.now())

# csv logs based on the time
csv_logger = CSVLogger('./logs/log_' + current_time + '.csv', append=True, separator=';')

# settings for reducing the learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = 1)

# Training the model on the training set, with early stopping using the validation set
callbacks = [EarlyStopping(patience = 5), reduce_lr, csv_logger]
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_data = (X_val, y_val), callbacks = callbacks)

# plot training history
# two plots
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'val'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'val'], loc='upper left')

fig.savefig('./logs/plot' + current_time + '.svg', format = 'svg')

# Evaluating the final model on the test set
y_pred = np.argmax(model.predict(X_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
