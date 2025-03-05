from saccade_extraction.pose import loadPoseEstimates, computePoseProjections
from matplotlib import pylab as plt
import numpy as np
import h5py
import pathlib as pl
import shutil

def validatePredictions(
    dlcFile,
    rsdFile,
    likelihoodThreshold=0.95,
    windowSize=500,
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
    if type(dlcFile) != pl.Path:
        dlcFile = pl.Path(dlcFile)
    targetDirectory = dlcFile.parent.joinpath('figures')
    if targetDirectory.exists():
        shutil.rmtree(targetDirectory)
    targetDirectory.mkdir()

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

    #
    for i, x1 in enumerate(range(0, t.size, windowSize)):
        x2 = x1 + windowSize
        ax.set_xlim([x1, x2])
        filename = f'epoch{i + 1}.png'
        fig.savefig(
            targetDirectory.joinpath(filename),
            dpi=300
        )
    fig.close()

    return fig, ax

def quanitfyPerformance(
    manualLabeling,
    modelPredictions,
    interframeIntervals,
    maximumLag=0.01
    ):
    """
    """

    framerate = round(1 / (np.median(np.loadtxt(interframeIntervals)) / 1000000000), 3)
    with open(manualLabeling, 'r') as stream:
        lines = stream.readlines()
    names = lines[0].rstrip('\n').split(',')

    # Exclude "special" saccades
    index = np.where([n == 'special' for n in names])[0].item()
    saccadeIndices = list()
    for saccadeIndex, ln in enumerate(lines[1:]):
        elements = ln.rstrip('\n').split(',')
        flag = bool(int(elements[index]))
        if flag:
            continue
        saccadeIndices.append(saccadeIndex)
    saccadeIndices = np.array(saccadeIndices)

    # Extract the frame indices for manually labeled saccades
    index = np.where([n == 'frame' for n in names])[0].item()
    frameIndicesTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        frameIndicesTrue.append(int(elements[index]))
    frameIndicesTrue = np.array(frameIndicesTrue)[saccadeIndices]

    # Extract the saccade labels for manually labeled saccades
    index = np.where([n == 'nasal' for n in names])[0].item()
    saccadeLabelsTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        saccadeLabelCoded = int(elements[index])
        saccadeLabel = 'n' if saccadeLabelCoded == 1 else 't'
        saccadeLabelsTrue.append(saccadeLabel)
    saccadeLabelsTrue = np.array(saccadeLabelsTrue)[saccadeIndices]

    # Extract frame indices and labels for predicted saccades
    with h5py.File(modelPredictions, 'r') as stream:
        frameIndicesPredicted = np.array(stream['saccade_onset']).ravel()
        saccadeLabelsPredicted = np.array([s.item().decode() for s in stream['saccade_labels']])

    # Compute the true positive rate
    matched = list()
    for t1, l1 in zip(frameIndicesTrue, saccadeLabelsTrue):

        # Compute time lag
        dt = t1 - frameIndicesPredicted
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesPredicted[closest]
        lag = abs(((t1 - t2) / framerate))

        #
        l2 = saccadeLabelsPredicted[closest]

        #
        if (lag <= maximumLag) and (l1 == l2):
            matched.append(True)
        else:
            matched.append(False)
    matched = np.array(matched)
    tpr = matched.sum() / matched.size

    # Compute the false positive rate
    unmatched = list()
    for t1, l1 in zip(frameIndicesPredicted, saccadeLabelsPredicted):

        #
        dt = t1 - frameIndicesTrue
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesTrue[closest]
        lag = abs(((t1 - t2) / framerate))

        #
        l2 = saccadeLabelsTrue[closest]

        #
        if (lag <= maximumLag) and (l1 == l2):
            unmatched.append(False)
        else:
            unmatched.append(True)
    unmatched = np.array(unmatched)
    fpr = unmatched.sum() / unmatched.size

    return tpr, fpr