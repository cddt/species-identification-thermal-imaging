import h5py    
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

"""
Discards videos missing a usable tag.
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

validation_num = 1500
test_num = 1500

f = h5py.File("/home/cddt/data-space/Cacophony/data1/dataset.hdf5", "r") # Read in the dataset
d = f[list(f.keys())[0]]                                        # Access the thermal videos key
clips = np.zeros([10664, 1, 3, 24, 24], dtype=np.float16)      # np.float16 saves storage space

def get_best_index(vid):
    """
    Returns an index such that the selected 45 frames from a given video correspond to
    the 45 frames where the animal is nearest to the camera.
    """
    mass = np.zeros(vid.attrs['frames'])
    for f in range(vid.attrs['frames']):
        mass[f] = np.sum(vid[str(f)][4])
#    total_mass_over_next_45 = np.cumsum(mass) - np.hstack([np.zeros(45), np.cumsum(mass[:-45])])
    return np.argmax(mass[::-1])

def make24x24(frame):
    """
    Interpolates a given frame so its largest dimension is 24. The padding uses the minimum
    of the frame's values across each channel.
    """
    scale = (24.5 / np.array(frame.shape[1:])).min()
    frame = torch.tensor(np.expand_dims(frame, 0))
    frame = np.array(nn.functional.interpolate(frame, scale_factor = scale, mode = 'area')[0])
    square = np.tile(np.min(frame, (1, 2)).reshape(3, 1, 1), (1, 24, 24))
    offset = ((np.array([24, 24]) - frame.shape[1:]) / 2).astype(np.int)
    square[:, offset[0] : offset[0]+frame.shape[1], offset[1] : offset[1]+frame.shape[2]] = frame
    return square

def normalize(frame):
    """
    Min-max normalizes the first channel (clipping outliers).
    Min-max normalizes the second channel for each frame independently.
    Min-max normalizes the third channel (clipping outliers).
    """
    frame[0] = np.clip((frame[0] - 2500) / 1000, 0, 1)
    frame[1] = np.nan_to_num((frame[1] - frame[1].min()) / (frame[1].max() - frame[1].min()))
    frame[2] = np.clip(frame[2] / 400, 0, 1)
    return frame

labels = []
processed = 0
for i in range(len(d.keys())):
    x = d[list(d.keys())[i]]
    for j in range(len(x.keys()) - 1):
        vid = x[list(x.keys())[j]]
        tag = vid.attrs['tag']
        if tag == "bird/kiwi":
            tag = "bird"
        if vid.attrs['frames'] >= 45 and not tag in ["unknown", "part", "poor tracking", "sealion"]:
            labels += [tag]
            ind = get_best_index(vid)
            frame = np.array(vid[str(ind)], dtype=np.float16)[:2]         # Read a single frame
            frame = np.concatenate([np.expand_dims(frame[0], 0), frame], 0) # The desired 3 channels
            frame = make24x24(frame)                                        # Interpolate the frame
            frame = normalize(frame)                                        # Normalizes each channel
            clips[processed, 0] = frame
            processed += 1                   
            if processed % 100 == 0:        
                print(processed, "clips processed!")

# We encode the labels as an integer for each class
labels = LabelEncoder().fit_transform(labels)

# We extract the training, test and validation sets, with a fixed random seed for reproducibility and stratification
clips, val_vids, labels, val_labels = train_test_split(clips, labels, test_size = validation_num, random_state = 123, stratify = labels)
train_vids, test_vids, train_labels, test_labels = train_test_split(clips, labels, test_size = test_num, random_state = 123, stratify = labels)

# We save all of the files
if not os.path.exists("./cacophony-preprocessed-single-frame"):
    os.makedirs("./cacophony-preprocessed-single-frame")
np.save("./cacophony-preprocessed-single-frame/training", train_vids)
np.save("./cacophony-preprocessed-single-frame/validation", val_vids)
np.save("./cacophony-preprocessed-single-frame/test", test_vids)
np.save("./cacophony-preprocessed-single-frame/training-labels", train_labels)
np.save("./cacophony-preprocessed-single-frame/validation-labels", val_labels)
np.save("./cacophony-preprocessed-single-frame/test-labels", test_labels)
