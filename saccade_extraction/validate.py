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