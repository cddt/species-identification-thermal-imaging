from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def load(name):
    X = np.load("./cacophony-preprocessed" + name + ".npy")
    y = np.load("./cacophony-preprocessed" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X[:, :, [5,7]], y_one_hot_encoded
print("horiz")
epochs = 50
batch_size = 32
learning_rate = 0.001

print("Dataset loading..", end = " ")
X_train_mvm, y_train = load("3/training")
X_val_mvm, y_val = load("3/validation")
X_test_mvm, y_test = load("3/test")
print("Dataset loaded!")
print(X_val_mvm.shape)

MLP = Sequential()
MLP.add(Dense(128, activation = "relu"))
MLP.add(Dense(13, activation="softmax"))

mvm_inputs = Input((45, 2))
mvm_features = LSTM(512, return_sequences=True)(mvm_inputs)
x = TimeDistributed(MLP)(mvm_features)
outputs = GlobalAveragePooling1D()(x)
model = Model(inputs=mvm_inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics=["accuracy"])
print(model.summary())

class DataGenerator(Sequence):
    def __init__(self, mvm, labels, batch_size, shuffle = True):
        self.mvm = mvm
        self.labels = labels
        self.indices = np.arange(mvm.shape[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.mvm[indices], self.labels[indices]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
train_data = DataGenerator(X_train_mvm, y_train, batch_size)
val_data = DataGenerator(X_val_mvm, y_val, batch_size)

reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 3, min_lr = 0.00001, verbose = 1)
checkpointer = ModelCheckpoint(filepath='justmvmmodel.hdf5', verbose=1, save_best_only = True, monitor = 'val_accuracy', mode = 'max')
callbacks = [EarlyStopping(patience = 10), reduce_lr, checkpointer]

model.fit(train_data,
          epochs = epochs,
          validation_data = val_data,
          callbacks = callbacks)

model.load_weights('justmvmmodel.hdf5')

y_pred = np.argmax(model.predict(X_test_mvm), axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
