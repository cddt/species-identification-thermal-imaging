from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime
from matplotlib import pyplot as plt
from keras.utils import plot_model

def load(name):
    X = np.load("./cacophony-preprocessed" + name + ".npy")
    y = np.load("./cacophony-preprocessed" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X, y_one_hot_encoded

epochs = 50
batch_size = 32
learning_rate = 0.001

print("Dataset loading..", end = " ")
# Loading the preprocessed videos
X_train, y_train = load("/training")
X_val, y_val = load("/validation")
X_test, y_test = load("/test")
# Since Keras likes the channels first data format
X_train = X_train.transpose(0,1,3,4,2)
X_val = X_val.transpose(0,1,3,4,2)
X_test = X_test.transpose(0,1,3,4,2)
# Loading the preprocessed movement features
X_train_mvm, _ = load("-movement/training")
X_val_mvm, _ = load("-movement/validation")
X_test_mvm, _ = load("-movement/test")
print("Dataset loaded!")

compactCNN = Sequential()
compactCNN.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(24,24,3)))
compactCNN.add(MaxPooling2D(pool_size=(2,2)))
compactCNN.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
compactCNN.add(MaxPooling2D(pool_size=(2,2)))
compactCNN.add(Flatten())
compactCNN.add(Dropout(0.5))
compactCNN.add(Dense(512, activation = "relu"))

MLP = Sequential()
MLP.add(Dense(128, activation = "relu"))
MLP.add(Dense(13, activation="softmax"))

vid_inputs = Input((45, 24, 24, 3))
mvm_inputs = Input((45, 9))
# CNN extracts 512 video features for each frame
vid_features = TimeDistributed(compactCNN)(vid_inputs)
# LSTM extracts 512 movement features for each frame
mvm_features = LSTM(512, return_sequences=True, dropout=0.5)(mvm_inputs)
# Concatenating for 1024 features for each frame
x = Concatenate()([vid_features, mvm_features])
# MLP makes a classification for each frame
x = TimeDistributed(MLP)(x)
# Outputting the mean classification of all frames
outputs = GlobalAveragePooling1D()(x)
model = Model(inputs=[vid_inputs, mvm_inputs], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])

# create log dir
if not os.path.exists("./logs"):
    os.makedirs("./logs")

current_time = str(datetime.datetime.now())

plot_model(model, to_file='./logs/model_' + current_time + '.png', show_shapes=True)

# csv logs based on the time
csv_logger = CSVLogger('./logs/log_' + current_time + '.csv', append=True, separator=';')

# settings for reducing the learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = 1)

# Training the model on the training set, with early stopping using the validation set
callbacks = [EarlyStopping(patience = 5), reduce_lr, csv_logger]
history = model.fit([X_train, X_train_mvm], y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle = True,
                    validation_data = ([X_val, X_val_mvm], y_val),
                    callbacks = callbacks)

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

# evalutate accuracy on hold out set
eval_metrics = model.evaluate([X_test, X_test_mvm], y_test, verbose = 1)
for idx, metric in enumerate(model.metrics_names):
    if metric == 'accuracy':
        print(metric + ' on hold out set:', round(100 * eval_metrics[idx], 1), "%", sep = "")

# Evaluating the final model on the test set
y_pred = np.argmax(model.predict([X_test, X_test_mvm]), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
