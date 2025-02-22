# saccade-extraction
This is a tool for the Felsen lab that extracts saccadic eye movements from
DeepLabCut pose estimates.

# Basic usage
Steps 1-3 describe how to create a project and train the models from scratch. If
you've already trained your models, go ahead and skip to step 4.

## 1. Initializing a new project 
The `initializeProject` function will make a new directory with a standard file structure and generate a config file. You will only need to do this once. The config file specifies the following parameters:
* `velocityThreshold` - Velocity threshold for peak detection (in percentiles)
* `minimumPeakDistance` - Minimum distance between putative saccades (in seconds) for peak detection
* `responseWindow` - Time window around saccades
* `nFeatures` - Number of time points for resampling
```Python
from saccade_extraction import (
    initializeProject,
    addNewSessions,
)
projectDirectory = '<Path to desired project directory>'
configFile = initializeProject(projectDirectory)
```

## 2. Collecting training data
The `addNewSessions` function processes the pose estimates and extracts high-velocity eye movements, i.e., putative saccades. You need to pass it a list of tuples in which each tuple contains the file path to the pose estimate output by DeepLabCut and the filepath to the file that stores the inter-frame intervals for that video recording. This only needs to be done if you intend to collect training data from a particular recording.
```Python
from saccade_extraction import addNewSessions
fileSets = [
    # First set of files
    ('<Path to first DeepLabCut pose estimate>',
     '<Path to first timestamps file>',
    ),
    # Second set of files
    ('<Path to second DeepLabCut pose estimate>',
     '<Path to second timestamps file>',
    ),
]
addNewSessions(fileSets)

```
Use the saccade labeling GUI to collect information about the putative saccades. For each putative saccade you wish to label, follow this workflow:
1. Open any of the folders you created by calling the `addNewSessions` function by clicking on the `Open` button.
2. Use the radio buttons on the left to indicate the direction of the saccade. Use the description on the y-axis of the top-left plot to judge whether the saccade moves the eye in the nasal or temporal direction. If you don't think the putative saccade is a real saccade, select the "Not a saccade" radio button.
3. Indicate when the saccade starts. Make sure you have enabled the `Start` line selector checkbox, then click anywhere on the Matplotlib figure. You should see the left vertical line turn green
4. Indicate when the saccade stops. Make sure you have enabled the `Stop` line selector checkbox, then click anywhere on the Matplotlib figure. You should see the right vertical line turn red.
5. Save your progress by clicking the `Save` button.
```Python
from saccade_extraction import labelPutativeSaccades
gui = labelPutativeSaccades(configFile)
```

## 3. Training the models
The `createTrainingDataset` function will collect all manually collected information and generate a dataset that will be used for training below. You only need to run this funciton if you have collected new training data since you last trained the models.
```Python
from saccade_extraction import createTrainingDataset
createTrainingDataset(configFile)
```
The `trainModels` function will use cross-validation to search a hyperparameter space for the unique combination of hyperparameters that result in the best classification and regression performance. This function accepts the keyword argument `trainingDatasetIndex` which specifies which training dataset to train on. The default behavior is for the models to train on the most recently generated training dataset, i.e., `trainingDatasetIndex=-1`. You only need to run this function if you have generated a new training dataset.
```Python
from saccade_extraction import trainModels
trainModels(configFile, trainingDatasetIndex=-1)
```

## 4. Extracting saccades
The `extractRealSaccades` function uses the models trained above to extract real saccades from the DeepLabCut pose estimates. It works much like the `addNewSessions` function in that you need to pass it a list of tuples where each tuple contains the path to a pose estimate and the path to the associated inter-frame intervals file. You can process as many recordings at once as you want. This function save an h5 file with the name `real_saccades_data.hdf` to the parent directory that contains the pose estimates. This file has the following datasets:
* `saccade_waveforms` (N saccades x M features) - The positional saccade waveforms for all real saccades (in pixels)
* `saccade_labels` (N saccades x 1) - The predicted saccade direction as single character, "N" for nasal saccades, or "T" for temporal saccades
* `saccade_labels_coded` (N saccades x 1) - The predicted saccade direction as a signed integer, +1 for nasal saccades, or -1 for temporal saccades
* `saccade_onset` (N saccades x 1) - The predicted onset of each saccade (in fractional frame indices)
* `saccade_offset` (N saccades x 1) - The predicted offset of each saccade (in fractional frame indices)
```Python
from saccade_extraction import extractRealSaccades
fileSets = [
    ('<Path to first DeepLabCut pose estimate>',
     '<Path to first timestamps file>',
    )
]
extractRealSaccades(configFile, fileSets)
```
Alternatively, you can use the command line interface (CLI) to extract saccades.
It only works for one set of files at a time, but it's easier to use. In the
terminal execute this command:
```Bash
saccade-extraction extract "<Path to config file>" "<Path to DeepLabCut pose estimate>" "<Path to timestamps file>"
```
