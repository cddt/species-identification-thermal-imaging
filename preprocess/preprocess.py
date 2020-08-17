import h5py    
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
import os

"""
Discards videos missing a tag.
Discards videos with fewer than 45 frames.
Trims the length of the videos to 45 frames.
Interpolates each of the cropped frames to 24 x 24.
Outputs 3 channels:
    (1) The raw thermal values (min-max normalization)
    (2) The raw thermal values (each frame normalized independently)
    (3) The thermal values minus the background (min-max normalization)
Splits the data into training, validation, and test sets.
Encodes the labels as integers.
Saves the pre-processed data and the labels as numpy arrays.
"""

validation_proportion = 0.15
test_proportion = 0.15
training_proportion = 1 - validation_proportion - test_proportion

np.random.seed(123)                                             # So the data splitting is reproducible
f = h5py.File("C://Users//hamis//Downloads//dataset.hdf5", "r") # Read in the dataset
d = f[list(f.keys())[0]]                                        # Access the thermal videos key
N = 10719                                                       # The number of videos not discarded
clips = np.zeros([N, 45, 3, 24, 24], dtype=np.float16)          # np.float16 saves storage space
labels = []                                                     # Each clip's label ("tag") is recorded

def get_best_index(vid):
    """
    Returns an index such that the selected 45 frames from a given video correspond to
    the 45 frames where the animal is nearest to the camera.
    """
    mass = np.zeros(vid.attrs['frames'])
    for f in range(vid.attrs['frames']):
        mass[f] = np.sum(vid[str(f)][4])
    total_mass_over_next_45 = np.cumsum(mass) - np.hstack([np.zeros(45), np.cumsum(mass[:-45])])
    return f - np.argmax(total_mass_over_next_45[::-1]) - 44

def interpolate(frame):
    """
    Interpolates a given frame so its largest dimension is 24. It does not force the
    frames to be squares.
    """
    scale = (24.5 / np.array(frame.shape[1:])).min()
    frame = torch.tensor(np.expand_dims(frame, 0))
    return np.array(nn.functional.interpolate(frame, scale_factor = scale, mode = 'area')[0])

def pad_horizontal(frame):
    """
    Makes tall rectangular frames square. The padding uses the frame's mean for the first
    and second channels, and zero for the third channel.
    """
    diff = 24 - frame.shape[1]
    left = int(diff / 2)
    right = diff - left
    pad = np.hstack([frame[0].mean(), frame[1].mean(), np.zeros(1, dtype=np.float16)])
    left = np.tile(pad, [left, 24, 1]).transpose([2, 0, 1])
    right = np.tile(pad, [right, 24, 1]).transpose([2, 0, 1])
    return np.concatenate([left, frame, right], 1)

def pad_vertical(frame):
    """
    Makes wide rectangular frames square. The padding uses the frame's mean for the first
    and second channels, and zero for the third channel.
    """
    diff = 24 - frame.shape[2]
    up = int(diff / 2)
    down = diff - up
    pad = np.hstack([frame[0].mean(), frame[1].mean(), np.zeros(1, dtype=np.float16)])
    up = np.tile(pad, [24, up, 1]).transpose([2, 0, 1])
    down = np.tile(pad, [24, down, 1]).transpose([2, 0, 1])
    return np.concatenate([up, frame, down], 2)

def normalize(frame):
    """
    Min-max normalizes the first channel (clipping outliers).
    Min-max normalizes the second channel for each frame independently.
    Min-max normalizes the third channel (clipping outliers).
    """
    frame[0] = np.clip((frame[0] - 2500) / 1000, 0, 1)
    frame[1] = np.nan_to_num((frame[1] - frame[1].min()) / (frame[1].max() - frame[1].min()))
    frame[2] = np.clip(frame[2] / 400 + 0.5, 0, 1)
    return frame

clips_processed = 0
for i in range(len(d.keys())):
    x = d[list(d.keys())[i]]
    for j in range(len(x.keys()) - 1):
        vid = x[list(x.keys())[j]]
        if vid.attrs['frames'] >= 45 and vid.attrs['tag'] != "unknown":
            labels += [vid.attrs['tag']]
            ind = get_best_index(vid)
            clip = np.zeros([45, 3, 24, 24])
            for f in range(ind, ind + 45):
                frame = np.array(vid[str(f)], dtype=np.float16)[:2]             # Read a single frame
                frame = np.concatenate([np.expand_dims(frame[0], 0), frame], 0) # The desired 3 channels
                frame = interpolate(frame)                                      # Interpolate the frame
                if frame.shape[1] < 24:                                         # Make the frame square
                    frame = pad_horizontal(frame)
                elif frame.shape[2] < 24:
                    frame = pad_vertical(frame)
                frame = normalize(frame)   # Normalizes each channel appropriately
                clip[f - ind] = frame      # Stores the processed frame
            clips[clips_processed] = clip  # Stores the processed clip
            clips_processed += 1           # Counts the processed clips
            if clips_processed % 100 == 0: # Prints updates to track progress
                print(clips_processed, "clips processed!")

# We shuffle the indices, and divide them into indices for the training, validation, and test sets.
inds = np.arange(N)
np.random.shuffle(inds)
training_size = round(N * training_proportion)
val_size = round(N * validation_proportion)
train_ind, val_ind, test_ind = np.split(inds, [training_size, training_size + val_size])

# We divide the videos according to the indices
train_vids = clips[train_ind]
val_vids = clips[val_ind]
test_vids = clips[test_ind]
del clips

# We divide the labels according to the indices
train_labels = [labels[i] for i in train_ind]
val_labels = [labels[i] for i in val_ind]
test_labels = [labels[i] for i in test_ind]

# We encode the labels as an integer for each class
encoder = LabelEncoder()
encoder.fit(labels)
train_labels = encoder.transform(train_labels)
val_labels = encoder.transform(val_labels)
test_labels = encoder.transform(test_labels)

# We save all of the files
if not os.path.exists("./cacophony-preprocessed"):
    os.makedirs("./cacophony-preprocessed")
np.save("./cacophony-preprocessed/training", train_vids)
np.save("./cacophony-preprocessed/validation", val_vids)
np.save("./cacophony-preprocessed/test", test_vids)
np.save("./cacophony-preprocessed/training-labels", train_labels)
np.save("./cacophony-preprocessed/validation-labels", val_labels)
np.save("./cacophony-preprocessed/test-labels", test_labels)
