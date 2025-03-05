from saccade_extraction.pose import loadPoseEstimates, computePoseProjections
from matplotlib import pylab as plt
import numpy as np
import h5py
import pathlib as pl
import shutil

def loadManualSaccadeLabeling(manualLabeling):
    """
    """

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
    frameIndicesTrue = np.array(frameIndicesTrue)[saccadeIndices].astype(np.float64)

    # Extract the saccade labels for manually labeled saccades
    index = np.where([n == 'nasal' for n in names])[0].item()
    saccadeLabelsTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        saccadeLabelCoded = int(elements[index])
        saccadeLabel = 'n' if saccadeLabelCoded == 1 else 't'
        saccadeLabelsTrue.append(saccadeLabel)
    saccadeLabelsTrue = np.array(saccadeLabelsTrue)[saccadeIndices]

    return frameIndicesTrue, saccadeLabelsTrue

def visualizePredictions(
    poseEstimates,
    saccadePredictions,
    manualLabeling=None,
    likelihoodThreshold=0.95,
    windowSize=500,
    ):
    """
    """

    pose, projections, uHorizontal, uVertical = computePoseProjections(
        poseEstimates,
        likelihoodThreshold
    )

    fig, ax = plt.subplots()
    t = np.arange(pose.shape[0])
    ax.plot(t, projections[:, 0], color='k')
    ylim = ax.get_ylim()
    
    #
    with h5py.File(saccadePredictions, 'r') as stream:
        saccadeLabels = np.array(stream['saccade_labels_coded']).ravel()
        saccadeOnsets = np.array(stream['saccade_onset']).ravel()
    if type(poseEstimates) != pl.Path:
        poseEstimates = pl.Path(poseEstimates)

    #
    targetDirectory = poseEstimates.parent.joinpath('figures')
    if targetDirectory.exists():
        shutil.rmtree(targetDirectory)
    targetDirectory.mkdir()

    ax.vlines(
        saccadeOnsets[saccadeLabels == -1],
        *ylim,
        color='r',
        alpha=0.5,
        label='Temporal (Pred)',
        lw=1
    )
    ax.vlines(
        saccadeOnsets[saccadeLabels == 1],
        *ylim,
        color='b',
        alpha=0.5,
        label='Nasal (Pred)',
        lw=1
    )
    ax.vlines(
        saccadeOnsets[saccadeLabels == 0],
        *ylim,
        color='0.5',
        alpha=0.5,
        label='NaS (Pred)',
        lw=1
    )
    
    #
    if manualLabeling is not None:
        frameIndicesTrue, saccadeLabelsTrue = loadManualSaccadeLabeling(
            manualLabeling
        )
        ax.vlines(
            frameIndicesTrue[saccadeLabelsTrue == 't'],
            *ylim,
            color='r',
            alpha=0.5,
            label='Temporal (True)',
            lw=1,
            linestyle=':'
        )
        ax.vlines(
            frameIndicesTrue[saccadeLabelsTrue == 'n'],
            *ylim,
            color='b',
            alpha=0.5,
            label='Nasal (True)',
            lw=1,
            linestyle=':'
        )

    #
    ax.set_ylabel('Nasal <-- Position (px) --> Temp.')
    ax.set_xlabel('Time (frames)')
    fig.legend()
    fig.tight_layout()
    ax.set_ylim(ylim)

    #
    for i, x1 in enumerate(range(0, t.size, windowSize)):
        x2 = x1 + windowSize
        ax.set_xlim([x1, x2])
        filename = f'epoch{i + 1}.png'
        fig.savefig(
            targetDirectory.joinpath(filename),
            dpi=300
        )
        fig.tight_layout()
    plt.close(fig)

    return fig, ax

def quantifyPerformance(
    manualLabeling,
    modelPredictions,
    interframeIntervals,
    maximumLag=0.015
    ):
    """
    """

    framerate = round(1 / (np.median(np.loadtxt(interframeIntervals)) / 1000000000), 3)
    frameIndicesTrue, saccadeLabelsTrue = loadManualSaccadeLabeling(manualLabeling)

    # Extract frame indices and labels for predicted saccades
    with h5py.File(modelPredictions, 'r') as stream:
        frameIndicesPredicted = np.array(stream['saccade_onset']).ravel()
        saccadeLabelsPredicted = np.array([s.item().decode() for s in stream['saccade_labels']])
    
    # Filter out NaS samples
    saccadeIndices = np.where(np.logical_or(
        saccadeLabelsPredicted == 'n',
        saccadeLabelsPredicted == 't',
    ))[0]
    frameIndicesPredicted = frameIndicesPredicted[saccadeIndices]
    saccadeLabelsPredicted = saccadeLabelsPredicted[saccadeIndices]

    #
    result = {
        'tpr': {
            'n': None,
            't': None,
        },
        'fpr': {
            'n': None,
            't': None,
        }
    }
    for saccadeLabel in ('n', 't'):

        # Compute the true positive rate
        matched = list()
        for saccadeIndex, t1 in enumerate(frameIndicesTrue):

            #
            if saccadeLabelsTrue[saccadeIndex] != saccadeLabel:
                continue

            # Compute time lag
            dt = frameIndicesPredicted - t1
            closest = np.argmin(np.abs(dt))
            t2 = frameIndicesPredicted[closest]
            lag = round(abs(((t1 - t2) / framerate)), 3)

            #
            if (lag <= maximumLag) and saccadeLabelsPredicted[closest] == saccadeLabel:
                matched.append(True)
            else:
                matched.append(False)
        matched = np.array(matched)
        tpr = round(matched.sum() / matched.size, 3)
        result['tpr'][saccadeLabel] = tpr

        # Compute the false positive rate
        unmatched = list()
        for saccadeIndex, t1 in enumerate(frameIndicesPredicted):

            #
            if saccadeLabelsPredicted[saccadeIndex] != saccadeLabel:
                continue

            #
            dt = frameIndicesTrue - t1
            closest = np.argmin(np.abs(dt))
            t2 = frameIndicesTrue[closest]
            lag = round(abs(((t1 - t2) / framerate)), 3)

            #
            if (lag <= maximumLag) and saccadeLabelsTrue[closest] == saccadeLabel:
                unmatched.append(False)
            else:
                unmatched.append(True)
        unmatched = np.array(unmatched)
        fpr = round(unmatched.sum() / unmatched.size, 3)
        result['fpr'][saccadeLabel] = fpr

    return result