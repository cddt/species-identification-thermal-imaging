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
Collates information of the movement of the cropped region.
Outputs 9 variables for each of the 45 frames:
    (1) Left boundary of cropped region
    (2) Upper boundary of cropped region
    (3) Right boundary of cropped region
    (4) Lower boundary of cropped region
    (5) Number of pixels above a temperature threshold (mass)
    (6) Cropped region horizontal velocity
    (7) Cropped region vertical velocity
    (8) Horizontal velocity scaled by area of cropped region
    (9) Vertical velocity scaled by area of cropped region
Normalizes each of the 9 variables.
Splits the data into training, validation, and test sets.
Encodes the labels as integers.
Saves the pre-processed data and the labels as numpy arrays.
"""

validation_num = 1500
test_num = 1500

f = h5py.File("C://Users//hamis//Downloads//dataset.hdf5", "r") # Read in the dataset
d = f[list(f.keys())[0]]                                        # Access the thermal videos key
clips = np.zeros([10664, 45, 9])

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
            try:
                b_h = vid.attrs['bounds_history'][ind : ind+45]
                m_h = vid.attrs['mass_history'][ind : ind+45]
                areas = (b_h[:,2] - b_h[:,0]) * (b_h[:,3] - b_h[:,1])
                centrex = (b_h[:,2] + b_h[:,0]) / 2
                centrey = (b_h[:,3] + b_h[:,1]) / 2
                xv = np.hstack((0, centrex[1:] - centrex[:-1]))
                yv = np.hstack((0, centrey[1:] - centrey[:-1]))
                axv = xv / areas**0.5
                ayv = yv / areas**0.5
                clips[processed] = np.hstack((b_h, np.vstack((m_h, xv, yv, axv, ayv)).T))
            except:
                print("Clip missing info, imputing with mean.")
                clips[processed] = np.tile([[76, 47, 96, 65, 180, 0.016, -0.015, 0.00076, -0.00055]], (45, 1))
                
            processed += 1                   
            if processed % 100 == 0:        
                print(processed, "clips processed!")
            
# Normalizing the data
clips -= np.mean(clips, (0,1))
clips /= np.std(clips, (0,1))

# We encode the labels as an integer for each class
labels = LabelEncoder().fit_transform(labels)

# We extract the training, test and validation sets, with a fixed random seed for reproducibility and stratification
clips, val_vids, labels, val_labels = train_test_split(clips, labels, test_size = validation_num, random_state = 123, stratify = labels)
train_vids, test_vids, train_labels, test_labels = train_test_split(clips, labels, test_size = test_num, random_state = 123, stratify = labels)

# We save all of the files
if not os.path.exists("./cacophony-preprocessed-movement"):
    os.makedirs("./cacophony-preprocessed-movement")
np.save("./cacophony-preprocessed-movement/training", train_vids)
np.save("./cacophony-preprocessed-movement/validation", val_vids)
np.save("./cacophony-preprocessed-movement/test", test_vids)
np.save("./cacophony-preprocessed-movement/training-labels", train_labels)
np.save("./cacophony-preprocessed-movement/validation-labels", val_labels)
np.save("./cacophony-preprocessed-movement/test-labels", test_labels)
