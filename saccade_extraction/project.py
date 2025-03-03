import yaml
import shutil
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks
from collections import OrderedDict
from saccade_extraction.pose import extractPutativeSaccades
from saccade_extraction.gui import launchGUI
from datetime import datetime
import pickle
import h5py

def initializeProject(
    targetDirectory
    ):
    """
    Initialize a new project
    """

    #
    if type(targetDirectory) != pl.Path:
        targetDirectory = pl.Path(targetDirectory)
    if targetDirectory.exists() == False:
        targetDirectory.mkdir()
    else:
        raise Exception('Target directory already exists: {targetDirectory}')

    #
    additionalDirectories = [
        ('models',),
        ('data',),
    ]
    for parts in additionalDirectories:
        path = targetDirectory.joinpath(*parts)
        if path.exists() == False:
            path.mkdir()

    # Default configuration file
    defaultSettings = {
        'projectDirectory': str(targetDirectory), # Project directory
        'velocityThreshold': 99, # Velocity threshold (as percentile)
        'minimumPeakDistance': 0.1, # Peak distance minimum (in seconds)
        'responseWindow': [-0.1, 0.1],
        'nFeatures': 50,
    }
    configFile = pl.Path(targetDirectory).joinpath('config.yaml')
    with open(configFile, 'w') as stream:
        yaml.dump(defaultSettings, stream, sort_keys=False)

    return configFile

def addNewSessions(
    configFile,
    fileSets,
    **kwargs
    ):
    """
    """

    #
    with open(configFile, 'r') as stream:
        config = yaml.safe_load(stream)
    projectDirectory = pl.Path(config['projectDirectory'])

    #
    for dlcFile, ifiFile in fileSets:

        #
        dlcFile, ifiFile = pl.Path(dlcFile), pl.Path(ifiFile)
        print(f'INFO: Extracting putative saccades from {dlcFile.name}')

        # Copy the pose estimates
        targetDirectory = projectDirectory.joinpath('data', dlcFile.stem)
        if targetDirectory.exists():
            print(f'WARNING: {dlcFile.name} is already a part of the training dataset')
            continue
        targetDirectory.mkdir()

        #
        try:

            #
            shutil.copy2(dlcFile, targetDirectory)
            shutil.copy2(ifiFile, targetDirectory)

            # Extract putative saccades
            putativeSaccadeWaveforms, frameIndices, frameTimestamps = extractPutativeSaccades(
                configFile,
                dlcFile,
                ifiFile,
                **kwargs
            )

            #
            nSaccades = putativeSaccadeWaveforms.shape[0]
            with h5py.File(targetDirectory.joinpath('putative_saccades_data.hdf'), 'w') as stream:
                stream.create_dataset(
                    'saccade_waveforms',
                    putativeSaccadeWaveforms.shape,
                    data=putativeSaccadeWaveforms,
                    dtype=putativeSaccadeWaveforms.dtype
                )
                stream.create_dataset(
                    'frame_indices',
                    frameIndices.shape,
                    data=frameIndices,
                    dtype=frameIndices.dtype
                )
                stream.create_dataset(
                    'frame_timestamps',
                    frameTimestamps.shape,
                    data=frameTimestamps,
                    dtype=frameTimestamps.dtype
                )
                stream.create_dataset(
                    'saccade_labels',
                    (nSaccades, 1),
                    data=np.full([nSaccades, 1], np.nan),
                    dtype=np.float32
                )
                stream.create_dataset(
                    'saccade_onset',
                    (nSaccades, 1),
                    data=np.full([nSaccades, 1], np.nan),
                    dtype=np.float32
                )
                stream.create_dataset(
                    'saccade_offset',
                    (nSaccades, 1),
                    data=np.full([nSaccades, 1], np.nan),
                    dtype=np.float32
                )

        # Handle errors
        except Exception as error:
            print(f'ERROR: Failed to add data from {dlcFile.name}')
            print(f'ERROR: {error}')
            shutil.rmtree(targetDirectory)

    return

def labelPutativeSaccades(
    configFile,
    seek=None
    ):
    """
    """

    gui = launchGUI(configFile, seek)

    return gui

def extractRealSaccades(
    configFile,
    fileSets,
    modelIndex=-1,
    **kwargs
    ):
    """
    """

    # Figure out which training dataset to use
    with open(configFile, 'r') as stream:
        configData = yaml.safe_load(stream)
    folders, timestamps = list(), list()
    for folder in pl.Path(configData['projectDirectory']).joinpath('models').iterdir():
        parts = folder.name.split('_')
        date, time = parts[0], parts[1]
        timestamp = datetime.strptime(f'{date}_{time}', '%Y-%m-%d_%H:%M:%S')
        folders.append(folder)
        timestamps.append(timestamp)

    sortedIndices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    sourceDirectory = folders[sortedIndices[modelIndex]]

    # Load the models
    with open(sourceDirectory.joinpath('classifier.pkl'), 'rb') as stream:
        clf = pickle.load(stream)
    with open(sourceDirectory.joinpath('regressor.pkl'), 'rb') as stream:
        reg = pickle.load(stream)

    #
    for dlcFile, ifiFile in fileSets:
        dlcFile = pl.Path(dlcFile)
        targetDirectory = dlcFile.parent
        putativeSaccadeWaveforms, frameIndices, frameTimestamps = extractPutativeSaccades(
            configFile,
            dlcFile,
            ifiFile,
            **kwargs
        )

        # Direction of saccades
        saccadeLabels = clf.predict(putativeSaccadeWaveforms[:, 0, :]).reshape(-1, 1)
        saccadeIndices = np.where(np.logical_or(
            saccadeLabels[:, 0] == -1,
            saccadeLabels[:, 0] ==  1,
        ))[0]
        nSaccades = saccadeIndices.size
        print(f'{nSaccades} real saccades extracted from {dlcFile.name}')

        # Timing of saccades (in frame indices)
        timeLags = reg.predict(putativeSaccadeWaveforms[:, 0, :])
        saccadeEpochs = list()
        for saccadeIndex in range(putativeSaccadeWaveforms.shape[0]):
            t = frameTimestamps[saccadeIndex]
            i = frameIndices[saccadeIndex]
            t0 = np.interp(
                (t.size - 1) / 2,
                np.arange(t.size),
                t
            )
            tStart = t0 + timeLags[saccadeIndex, 0]
            tStop = t0 + timeLags[saccadeIndex, 1]
            iStart = np.interp(tStart, t, i)
            iStop = np.interp(tStop, t, i)
            saccadeEpoch = np.array([iStart, iStop])
            saccadeEpochs.append(saccadeEpoch)
        saccadeEpochs = np.around(np.array(saccadeEpochs), 3)

        # Save the results
        fp = targetDirectory.joinpath(f'{dlcFile.stem}_saccades.hdf')
        realSaccadeWaveforms = putativeSaccadeWaveforms[saccadeIndices, 0, :]
        realSaccadeEpochs = saccadeEpochs[saccadeIndices, :]
        realSaccadeLabels = saccadeLabels[saccadeIndices, :]
        with h5py.File(fp, 'w') as stream:
            ds = stream.create_dataset(
                'saccade_waveforms',
                shape=realSaccadeWaveforms.shape,
                dtype=realSaccadeWaveforms.dtype,
                data=realSaccadeWaveforms
            )
            ds = stream.create_dataset(
                'saccade_onset',
                shape=(nSaccades, 1),
                dtype=realSaccadeEpochs.dtype,
                data=realSaccadeEpochs[:, 0].reshape(-1, 1)
            )
            ds = stream.create_dataset(
                'saccade_offset',
                shape=(nSaccades, 1),
                dtype=realSaccadeEpochs.dtype,
                data=realSaccadeEpochs[:, 1].reshape(-1, 1)
            )
            ds = stream.create_dataset(
                'saccade_labels_coded',
                shape=(nSaccades, 1),
                dtype=realSaccadeLabels.dtype,
                data=realSaccadeLabels.reshape(-1, 1)
            )
            ds = stream.create_dataset(
                'saccade_labels',
                shape=(nSaccades, 1),
                dtype=h5py.special_dtype(vlen=str),
                data=np.array(['N' if l == 1 else 'T' for l in realSaccadeLabels], dtype=object).reshape(-1, 1)
            )   

    return