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
    ax.vlines(
        saccadeOnsets[saccadeLabels == 0],
        *ylim,
        color='0.5',
        alpha=0.25,
        label='NaS'
    )
    fig.legend()

    return fig, ax

def compareWithManualLabeling(
    csvFile,
    rsdFile,
    framerate,
    frameColumn='frame',
    labelColumn='nasal_temporal',
    maximumLag=0.015
    ):
    """
    """

    with open(csvFile, 'r') as stream:
        lines = stream.readlines()
    names = lines[0].rstrip('\n').split(',')
    index = np.where([n == frameColumn for n in names])[0].item()
    frameIndicesTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        frameIndicesTrue.append(int(elements[index]))
    frameIndicesTrue = np.array(frameIndicesTrue)

    #
    index = np.where([n == labelColumn for n in names])[0].item()
    saccadeLabelsTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        saccadeLabelCoded = int(elements[index])
        saccadeLabel = 'N' if saccadeLabelCoded == -1 else 'T'
        saccadeLabelsTrue.append(saccadeLabel)
    saccadeLabelsTrue = np.array(saccadeLabelsTrue)

    #
    with h5py.File(rsdFile, 'r') as stream:
        frameIndicesPredicted = np.array(stream['saccade_onset']).ravel()
        saccadeLabelsPredicted = np.array([s.item().decode() for s in stream['saccade_labels']])

    # Compute the true positive rate
    matched = list()
    lags = list()
    for t1, l1 in zip(frameIndicesTrue, saccadeLabelsTrue):

        # Compute time lag
        dt = t1 - frameIndicesPredicted
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesPredicted[closest]
        lag = abs(((t1 - t2) / framerate))
        lags.append(lag)

        #
        l2 = saccadeLabelsPredicted[closest]

        #
        if (lag <= maximumLag) and (l1 == l2):
            matched.append(True)
        else:
            matched.append(False)
    matched = np.array(matched)
    lags = np.array(lags)
    tpr = matched.sum() / matched.size

    # TODO: Compute the false positive rate
    fpr = None

    return tpr, fpr, lags, frameIndicesTrue, frameIndicesPredicted