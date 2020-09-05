# Species Identification in Thermal Imaging
Documentation for COMPSCI 760 group project

## 

## Preprocessing 

The main [preprocessing code](preprocess/preprocess.py) loads the raw dataset.hdf5 file and performs the following operations: 
  - Discards videos missing a usable tag.
  - Discards videos with fewer than 45 frames.
  - Trims the length of the videos to 45 frames.
  - Interpolates each of the cropped frames to 24 x 24.
  - Outputs 3 channels:

    - The raw thermal values (min-max normalization)
    - The raw thermal values (each frame normalized independently)
    - The thermal values minus the background (min-max normalization)

  - Splits the data into training, validation, and test sets. This is performed using a fixed seed for reperformability, and the size of the split is 7664/1500/1500 (72%/14%/14%). Stratification is using in the split to ensure classes are equally represented across the data sets. 
  - Encodes the labels as integers.
  - Saves the pre-processed data and the labels as numpy arrays.

The [single-frame preprocessing code](preprocess/preprocess-single-frame.py) performs the same operations as the main preprocessing code, except it extracts only the most useful single frame from the entire video clip. 

The [movement preprocessing code](preprocess/preprocess-movement.py) collates information of the movement of the cropped region, and outputs 9 normalised variables for each of the 45 frames:
  1) Left boundary of cropped region
  2) Upper boundary of cropped region
  3) Right boundary of cropped region
  4) Lower boundary of cropped region
  5) Number of pixels above a temperature threshold (mass)
  6) Cropped region horizontal velocity
  7) Cropped region vertical velocity
  8) Horizontal velocity scaled by area of cropped region
  9) Vertical velocity scaled by area of cropped region

## Data Augmentation

### Video augmentation 

### Movement data

## Model Design

### Using well-known image recognition architectures with and without pre-training

#### ResNet-18 without pre-trained model

#### ResNet-18 with pre-trained model (ImageNet)

#### ResNet-50 without pre-trained model

#### ResNet-50 with pre-trained model (ImageNet)

### Using image processing model architectures

#### Conv3D

#### ConvLSTM

#### R2.1D

### Neural Architecture Search

## Hidden / Latent Space Visualisation 

The [dimentionality reduction code](preprocess/dim_reduction.py) allows any selected layer from any model to be visualised by calling `plot_dim_reduction(model, layer)`

## Hyperparameter Tuning

## Results

