from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime
from matplotlib import pyplot as plt
import argparse

from ray.tune.integration.keras import TuneReporterCallback
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

def load(name):
    X = np.load("./cacophony-preprocessed" + name + ".npy")
    y = np.load("./cacophony-preprocessed" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return pin_in_object_store(X), pin_in_object_store(y_one_hot_encoded)

def train_model(config):

    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Since Keras likes the channels first data format    
    X_train = np.asarray(get_pinned_object(X_train2))
    X_train = X_train.transpose(0,1,3,4,2)
    X_test = np.asarray(get_pinned_object(X_test2))
    X_test = X_test.transpose(0,1,3,4,2)
    X_val = np.asarray(get_pinned_object(X_val2))
    X_val = X_val.transpose(0,1,3,4,2)
    y_train = np.asarray(get_pinned_object(y_train2))
    y_test = np.asarray(get_pinned_object(y_test2))
    y_val = np.asarray(get_pinned_object(y_val2))

    X_train_mvm = np.asarray(get_pinned_object(X_train_mvm2))
    X_test_mvm = np.asarray(get_pinned_object(X_test_mvm2))
    X_val_mvm = np.asarray(get_pinned_object(X_val_mvm2))

    compactCNN = Sequential()
    compactCNN.add(Conv2D(config['conv2d_1'], kernel_size=(3,3), activation="relu", input_shape=(24,24,3)))
    compactCNN.add(MaxPooling2D(pool_size=(2,2)))
    compactCNN.add(Conv2D(config['conv2d_2'], kernel_size=(3,3), activation="relu"))
    compactCNN.add(MaxPooling2D(pool_size=(2,2)))
    compactCNN.add(Flatten())
    compactCNN.add(Dropout(config['dropout_1']))
    compactCNN.add(Dense(config['dense_1'], activation = "relu"))

    MLP = Sequential()
    MLP.add(Dense(config['dense_2'], activation = "relu"))
    MLP.add(Dense(13, activation="softmax"))

    vid_inputs = Input((45, 24, 24, 3))
    mvm_inputs = Input((45, 9))
    # CNN extracts 512 video features for each frame
    vid_features = TimeDistributed(compactCNN)(vid_inputs)
    # LSTM extracts 512 movement features for each frame
    mvm_features = LSTM(config['lstm_1'], return_sequences=True, dropout=config['dropout_2'])(mvm_inputs)
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

    # csv logs based on the time
    csv_logger = CSVLogger('./logs/log_' + current_time + '.csv', append=True, separator=';')

    # settings for reducing the learning rate
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = 1)

    # Training the model on the training set, with early stopping using the validation set
    callbacks = [EarlyStopping(patience = 5), reduce_lr, csv_logger, TuneReporterCallback()]
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

if __name__ == '__main__':
    import ray
    from ray import tune
    #from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.utils import pin_in_object_store, get_pinned_object

    ray.init(num_cpus=4 if args.smoke_test else None, redis_max_memory=10**9)

    print("Dataset loading..", end = " ")
    # Loading the preprocessed videos
    X_train2, y_train2 = load("/training")
    X_val2, y_val2 = load("/validation")
    X_test2, y_test2 = load("/test")
    # Loading the preprocessed movement features
    X_train_mvm2, _ = load("-movement/training")
    X_val_mvm2, _ = load("-movement/validation")
    X_test_mvm2, _ = load("-movement/test")
    print("Dataset loaded!")

    space = {
            "conv2d_1": hp.choice("conv2d_1", [16, 32, 64, 128, 256]),
            "conv2d_2": hp.choice("conv2d_2", [16, 32, 64, 128, 256]),
            "dense_1": hp.choice("dense_1", [64, 128, 256, 512, 768]),
            "dense_2": hp.choice("dense_2", [64, 128, 256, 512, 768]),
            "lstm_1": hp.choice("lstm_1", [32, 64, 128, 256, 512, 768, 1024]),
            "dropout_1": hp.uniform("dropout_1", 0.1, 0.7),
            "dropout_2": hp.uniform("dropout_2", 0.1, 0.7)
            }

    current_best_params = [{"conv2d_1": 32,
            "conv2d_2": 64,
            "dense_1": 512,
            "dense_2": 128,
            "lstm_1": 512,
            "dropout_1": 0.5,
            "dropout_2": 0.5
            }]

    hyperopt = HyperOptSearch(space, metric="mean_accuracy", mode="min")#, points_to_evaluate=current_best_params)

    tune.run(
        train_model,
        #name="exp",
        #scheduler=sched,
        search_alg=hyperopt,
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 5 if args.smoke_test else 12000
        },
        num_samples=3,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        })

