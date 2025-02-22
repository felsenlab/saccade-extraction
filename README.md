# saccade-extraction
This is a tool for the Felsen lab that extracts saccadic eye movements from
DeepLabCut pose estimates.

# Basic usage

## Initializing a project 
Import required functions.
```Python
from saccade_extraction import (
    initializeProject,
    addNewSessions,
)
```
The `initializeProject` function will make a new directory with a standard file structure and generate a config file. You will only need to do this once. The config file specifies the following parameters:
* `velocityThreshold` - Velocity threshold for peak detection (in percentiles)
* `minimumPeakDistance` - Minimum distance between putative saccades (in seconds) for peak detection
* `responseWindow` - Time window around saccades
* `nFeatures` - Number of time points for resampling
```Python
projectDirectory = '/home/josh/Desktop/my_cool_project_2'
configFile = initializeProject(projectDirectory)
```
