# saccade-extraction
This is a tool for the Felsen lab that extracts saccadic eye movements from
DeepLabCut pose estimates.

# Basic usage

## Initializing a new project 
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

## Collecting training data
The `addNewSessions` function processes the pose estimates and extracts high-velocity eye movements, i.e., putative saccades. You need to pass it a list of tuples in which each tuple contains the file path to the pose estimate output by DeepLabCut and the filepath to the file that stores the inter-frame intervals for that video recording. This only needs to be done if you intend to collect training data from a particular recording.
```Python
from saccade_extraction import addNewSessions, labelPutativeSaccades
fileSets = [
    # First set of files
    ('<Path to first DeepLabCut pose estimates>',
     '<Path to first timestamps file>',
    ),
    # Second set of files
    ('<Path to second DeepLabCut pose estimates>',
     '<Path to second timestamps file>',
    ),
]

```
Use the saccade labeling GUI to collect information about the putative saccades. For each putative saccade you wish to label, follow this workflow:
1. Open any of the folders you created by calling the `addNewSessions` function by clicking on the `Open` button.
2. Use the radio buttons on the left to indicate the direction of the saccade. Use the description on the y-axis of the top-left plot to judge whether the saccade moves the eye in the nasal or temporal direction. If you don't think the putative saccade is a real saccade, select the "Not a saccade" radio button.
3. Indicate when the saccade starts. Make sure you have enabled the `Start` line selector checkbox, then click anywhere on the Matplotlib figure. You should see the left vertical line turn green
4. Indicate when the saccade stops. Make sure you have enabled the `Stop` line selector checkbox, then click anywhere on the Matplotlib figure. You should see the right vertical line turn red.
5. Save your progress by clicking the `Save` button.
```Python
gui = labelPutativeSaccades(configFile)
```

## Training the models

## Extracting saccades