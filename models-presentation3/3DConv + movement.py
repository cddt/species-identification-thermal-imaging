from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import numpy as np
import tensorflow as tf
import random as python_random
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime
from matplotlib import pyplot as plt

"""
Predicts the class using 3DConv.
"""

# set seeds
np.random.seed(7654)
python_random.seed(7654)
tf.random.set_seed(7654)

def load(name):
    # Loads the videos and converts the labels into one-hot encoding for Keras
    X = np.load("/home/cddt/data-space/COMPSCI760/cacophony-preprocessed" + name + ".npy")
    y = np.load("/home/cddt/data-space/COMPSCI760/cacophony-preprocessed" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X, y_one_hot_encoded

def define_model_3DConv():

    c3d = Sequential()
    c3d.add(Conv3D(32, (3,3,3), activation='relu', input_shape = (45,24,24,3)))
    c3d.add(MaxPooling3D(pool_size=(2,2,2)))
    c3d.add(Conv3D(64, kernel_size=(3,3,3), activation='relu'))
    c3d.add(MaxPooling3D(pool_size=(2,2,2)))
    c3d.add(Flatten())
    c3d.add(Dropout(0.5))
    c3d.add(Dense(1024, activation='relu'))

    MLP = Sequential()
    MLP.add(Dense(128, activation = "relu"))
    MLP.add(Dense(13, activation="softmax"))

    vid_inputs = Input((45, 24, 24, 3))
    mvm_inputs = Input((45, 9))
    # CNN extracts 512 video features for each frame
    vid_features = TimeDistributed(c3d)(vid_inputs)
    # LSTM extracts 512 movement features for each frame
    mvm_features = LSTM(512, return_sequences=True, dropout = 0.2, recurrent_dropout = 0.2)(mvm_inputs)
    # Concatenating for 1024 features for each frame
    x = Concatenate()([vid_features, mvm_features])
    # LSTM across both image and movement data
    x = LSTM(512, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2)(x)
    # MLP makes a classification for each frame
    x = TimeDistributed(MLP)(x)
    # Outputting the mean classification of all frames
    outputs = GlobalAveragePooling1D()(x)
    model = Model(inputs=[vid_inputs, mvm_inputs], outputs=outputs)      

    return model

class DataGenerator(Sequence):
    def __init__(self, vids, mvm, labels, batch_size, flip = False, angle = 0, crop = 0, shift = 0, shuffle = True):
        self.vids = vids
        self.mvm = mvm
        self.labels = labels
        self.indices = np.arange(vids.shape[0])
        self.batch_size = batch_size
        self.flip = flip
        self.angle = angle
        self.crop = crop
        self.shift = shift
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def random_zoom(self, batch, x, y):
        ax = np.random.uniform(self.crop)
        bx = np.random.uniform(ax)
        ay = np.random.uniform(self.crop)
        by = np.random.uniform(ay)
        x = x*(1-ax/batch.shape[2]) + bx
        y = y*(1-ay/batch.shape[3]) + by
        return x, y

    def random_rotate(self, batch, x, y):
        rad = np.random.uniform(-self.angle, self.angle)/180*np.pi
        rotm = np.array([[np.cos(rad),  np.sin(rad)],
                         [-np.sin(rad), np.cos(rad)]])
        xm, ym = x.mean(), y.mean()
        x, y = np.einsum('ji, mni -> jmn', rotm, np.dstack([x-xm, y-ym]))
        return x+xm, y+ym

    def random_translate(self, batch, x, y):
        xs = np.random.uniform(-self.shift, self.shift)
        ys = np.random.uniform(-self.shift, self.shift)
        return x + xs, y + ys

    def horizontal_flip(self, batch):
        return np.flip(batch, 3)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        vids = np.array(self.vids[indices])
        x, y = np.meshgrid(range(vids.shape[2]), range(vids.shape[3]))
        if self.crop:
            x, y = self.random_zoom(vids, x, y)
        if self.angle:
            x, y = self.random_rotate(vids, x, y)
        if self.shift:
            x, y = self.random_translate(vids, x, y)
        if self.flip and np.random.random() < 0.5:
            vids = self.horizontal_flip(vids)
        x = np.clip(x, 0, vids.shape[2]-1).astype(np.int)
        y = np.clip(y, 0, vids.shape[3]-1).astype(np.int)
        vids = vids[:,:,x,y].transpose(0,1,3,2,4)
        if self.mvm is not None:
            out = [vids, self.mvm[indices]], self.labels[indices]
        else:
            out = vids, self.labels[indices]
        return out
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

epochs = 100
batch_size = 32
learning_rate = 0.001

# Loads the preprocessed datasets
print("Dataset loading..", end = " ")
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

# create log dir
if not os.path.exists("./logs/3DConv"):
    os.makedirs("./logs/3DConv")

current_time = str(datetime.datetime.now())

# csv logs based on the time
csv_logger = CSVLogger('./logs/3DConv/log_' + current_time + '.csv', append=True, separator=';')

# settings for reducing the learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 3, min_lr = 0.00001, verbose = 1)

# save the model at the best epoch
checkpointer = ModelCheckpoint(filepath='./logs/3DConv/best_model_' + current_time + '.hdf5', verbose=1, save_best_only = True, monitor = 'val_accuracy', mode = 'max')

# Training the model on the training set, with early stopping using the validation set
callbacks = [EarlyStopping(patience = 10), reduce_lr, csv_logger, checkpointer]

train_data = DataGenerator(X_train, X_train_mvm, y_train, batch_size, True, 0, 0, 0)
val_data = DataGenerator(X_val, X_val_mvm, y_val, batch_size)

model = define_model_3DConv()

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])

print(model.summary())

history = model.fit(train_data, epochs = epochs, batch_size = batch_size, shuffle = True, validation_data = val_data, callbacks = callbacks, verbose = 2)

# plot training history
# two plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,12))

fig.patch.set_facecolor('white')

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

fig.savefig('./logs/3DConv/plot' + current_time + '.svg', format = 'svg')

model.load_weights('./logs/3DConv/best_model_' + current_time + '.hdf5')

# evalutate accuracy on hold out set
eval_metrics = model.evaluate(X_test, X_test_mvm, y_test, verbose = 0)
for idx, metric in enumerate(model.metrics_names):
    if metric == 'accuracy':
        print(metric + ' on hold out set:', round(100 * eval_metrics[idx], 1), "%", sep = "")

# Evaluating the final model on the test set
y_pred = np.argmax(model.predict(X_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


