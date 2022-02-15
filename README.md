# COLENET: Baseline model for the Cholec80-CSV dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository holds the data preprocessing and the baseline model training and testing 
scripts for the dataset Cholec80-CSV, the first open dataset for Strasgerg's Criteria Detection.


## USAGE

### Download Cholec80-CVS dataset

Download Cholec80-CSV from [its official repository](https://duckduckgo.com).  
Then store it on `data/` with the name `surgeons_annotations.xlsx`

### Configuration file

To ease the use of this repository we provide a configuration file. It must be filled before running
any of the provided scripts.

- `dataset_path`: Absolute path of CHOLEC80 dataset.
- `frames_path`: Path where frames will be stored.

### Data Wrangling

All the scripts inside the folder `data_preprocessing` transform the raw data into usable data formats.
These scripts handle both the annotations provided by the surgeons and also the videos of the CHOLEC80 dataset. 
They must be executed from the `data_preprocessing` folder and must be executed in the following order:

- get_valid_frames.py
- video_2_frames.py
- annotation_2_labels.py
- get_training_sets.py

After that, the relevant frames of each video will be stored in the path specified on the config file,
and also the training, validation and testing sets will be created on the `data` folder.

### Model training

##### Regular training

To train any of the supported backbones simply run `train.py`. In this file, you can specify 
the backbone, and the log directory name. The script will automatically create a directory and store 
the training logs, and the best model weights.
The supported backbones are:

`vgg`
`resnet`
`alexnet`
`densenet`
`inception`

##### Folds Training

If desired, it is possible to execute k-fold cross validation by executing the 
script `train_folds.py` as described in the dataset description paper.  This script, 
automatically load the file `data/folds.json` which
stores the splits used to generate the results reported.


### Files Description

#### data_preprocessing/video_2_frames.py

This script divides the CHOLEC80 videos into separated frames, these frames are stored
on the location specified on the config file. This procedure is executed on the frames
that occurs before the Clipping and Cutting Phase.

#### data_preprocessing/get_valid_frames.py

This script extracts the index of the last valid frame, which corresponds with the last frame 
before the Clipping and Cutting Phase.

#### data_preprocessing/annotations_to_labels.py

This script transforms the annotations provided by the surgeons into a format easier to use in subsequent
steps. It stores the annotations per video in the `data/surgeons_annotations` folder.


