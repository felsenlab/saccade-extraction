from saccade_extraction.pose import loadPoseEstimates, computePoseProjections
from matplotlib import pylab as plt
import numpy as np
import h5py

def validatePredictions(
    dlcFile,
    rsdFile,
    likelihoodThreshold=0.95,
    ):
    """
    """

    pose, projections, uHorizontal, uVertical = computePoseProjections(
        dlcFile,
        likelihoodThreshold
    )

    fig, ax = plt.subplots()
    t = np.arange(pose.shape[0])
    ax.plot(t, projections[:, 0], color='k')
    ylim = ax.get_ylim()
    
    #
    with h5py.File(rsdFile, 'r') as stream:
        saccadeLabels = np.array(stream['saccade_labels_coded']).ravel()
        saccadeOnsets = np.array(stream['saccade_onset']).ravel()

    ax.vlines(
        saccadeOnsets[saccadeLabels == -1],
        *ylim,
        color='r',
        alpha=0.25,
        label='Temporal saccades'
    )
    ax.vlines(
        saccadeOnsets[saccadeLabels == 1],
        *ylim,
        color='b',
        alpha=0.25,
        label='Nasal saccades'
    )
    fig.legend()

    return fig, ax

def compareWithManualLabeling(
    csvFile,
    rsdFile,
    framerate,
    key='frame',
    maximumDistance=0.015
    ):
    """
    """

    with open(csvFile, 'r') as stream:
        lines = stream.readlines()
    names = lines[0].rstrip('\n').split(',')
    index = np.where([n == key for n in names])[0].item()
    frameIndicesTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        frameIndicesTrue.append(int(elements[index]))
    frameIndicesTrue = np.array(frameIndicesTrue)

    #
    with h5py.File(rsdFile, 'r') as stream:
        frameIndicesPredicted = np.array(stream['frame_indices'])

    #
    matched = list()
    for t1 in frameIndicesTrue:
        dt = t1 - frameIndicesPredicted
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesPredicted[closest]
        if ((t1 - t2) / framerate) < maximumDistance:
            matched.append(True)
        else:
            matched.append(False)
    matched = np.array(matched)

    return matched.sum() / matched.size