from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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
X_train_mvm, _ = load("3/training")
X_val_mvm, _ = load("3/validation")
X_test_mvm, _ = load("3/test")
print("Dataset loaded!")

compactCNN = Sequential()
compactCNN.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(24,24,3)))
compactCNN.add(MaxPooling2D(pool_size=(2,2)))
compactCNN.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
compactCNN.add(MaxPooling2D(pool_size=(2,2)))

MLP = Sequential()
MLP.add(Dropout(0.5))
MLP.add(Dense(512, activation = "relu"))
MLP.add(Dense(13, activation="softmax"))

vid_inputs = Input((45, 24, 24, 3))
mvm_inputs = Input((45, 9))
vid_features = TimeDistributed(compactCNN)(vid_inputs)
vid_features = MaxPooling3D(pool_size=(45, 1, 1))(vid_features)
vid_features = Flatten()(vid_features)
mvm_features = LSTM(512, return_sequences=False, dropout=0.5)(mvm_inputs)
x = Concatenate()([vid_features, mvm_features])
x = MLP(x)
model = Model(inputs=[vid_inputs, mvm_inputs], outputs=x)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])
print(model.summary())

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

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = 1)
callbacks = [reduce_lr]
model.fit(train_data,
          epochs = epochs,
          validation_data = val_data,
          callbacks = callbacks)

# Evaluating the final model on the test set
y_pred = np.argmax(model.predict([X_test, X_test_mvm]), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
