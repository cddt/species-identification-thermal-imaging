from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

"""
Predicts the class using ConvLSTM.
https://thebinarynotes.com/video-classification-keras-convlstm/
"""

def load(name):
    # Loads the videos and converts the labels into one-hot encoding for Keras
    X = np.load("./cacophony-preprocessed/" + name + ".npy")
    y = np.load("./cacophony-preprocessed/" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], 17])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X, y_one_hot_encoded

epochs = 10
batch_size = 10
learning_rate = 0.001

# Loads the preprocessed datasets
print("Dataset loading..", end = " ")
X_train, y_train = load("training")
X_val, y_val = load("validation")
X_test, y_test = load("test")
print("Dataset loaded!")

model = Sequential()
for _ in range(4):
    model.add(ConvLSTM2D(64, (3,3), return_sequences = True, data_format = "channels_first", input_shape = (45,3,24,24)))
    model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(17, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])

# Training the model on the training set, with early stopping using the validation set
callbacks = [EarlyStopping(patience = 7)]
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_data = (X_val, y_val), callbacks = callbacks)

# Evaluating the final model on the test set
y_pred = np.argmax(model.predict(X_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
