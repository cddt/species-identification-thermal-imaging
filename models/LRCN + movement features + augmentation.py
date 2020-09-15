from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence, plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime
from matplotlib import pyplot as plt

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
        x, y = np.einsum('ji, mni -> jmn', rotm, np.dstack([x, y]))
        return x, y

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
        return [vids, self.mvm[indices]], self.labels[indices]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
train_data = DataGenerator(X_train, X_train_mvm, y_train, batch_size, True, 10, 0, 0)
val_data = DataGenerator(X_val, X_val_mvm, y_val, batch_size)

# create log dir
if not os.path.exists("./logs"):
    os.makedirs("./logs")

current_time = str(datetime.datetime.now())

plot_model(model, to_file='./logs/model_' + current_time + '.png', show_shapes=True)

# csv logs based on the time
csv_logger = CSVLogger('./logs/log_' + current_time + '.csv', append=True, separator=';')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.00001, verbose = 1)
callbacks = [EarlyStopping(patience = 5), reduce_lr, csv_logger]
history = model.fit(train_data,
          epochs = epochs,
          validation_data = val_data,
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
