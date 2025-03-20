from saccade_extraction.pose import _loadPoseEstimates, _computeProjections
from matplotlib import pylab as plt
import numpy as np
import h5py
import pathlib as pl
import shutil
from scipy.signal import find_peaks

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
        frameIndicesTrue.append(float(elements[index]))
    frameIndicesTrue = np.array(frameIndicesTrue)[saccadeIndices].astype(np.float64)

    # Extract the saccade labels for manually labeled saccades
    index = np.where([n == 'nasal' for n in names])[0].item()
    saccadeLabelsTrue = list()
    for ln in lines[1:]:
        elements = ln.rstrip('\n').split(',')
        saccadeLabelCoded = int(elements[index])
        if saccadeLabelCoded == -1:
            saccadeLabel = 't'
        elif saccadeLabelCoded == 1:
            saccadeLabel = 'n'
        else:
            saccadeLabel = '-'
        saccadeLabelsTrue.append(saccadeLabel)
    saccadeLabelsTrue = np.array(saccadeLabelsTrue)[saccadeIndices]

    return frameIndicesTrue, saccadeLabelsTrue

def visualizePredictions(
    poseEstimates,
    saccadePredictions,
    manualLabeling=None,
    likelihoodThreshold=0.95,
    epochSize=500,
    generateEpochImages=False,
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
    if generateEpochImages:
        for i, x1 in enumerate(range(0, t.size, epochSize)):
            x2 = x1 + epochSize
            ax.set_xlim([x1, x2])
            filename = f'epoch{i + 1}.png'
            fig.savefig(
                targetDirectory.joinpath(filename),
                dpi=300
            )
            fig.tight_layout()

    return fig, ax

def _computeOutcomeRates(
    frameIndicesTrue,
    frameIndicesPredicted,
    saccadeLabelsTrue,
    saccadeLabelsPredicted,
    maximumLag,
    fps,
    ):
    """
    """

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
    for saccadeLabel, saccadeLabelCoded in zip(['n', 't'], [1, -1]):

        # Compute the true positive rate
        matched = list()
        for saccadeIndex, t1 in enumerate(frameIndicesTrue):

            # Skip non-target samples
            if saccadeLabelsTrue[saccadeIndex] != saccadeLabelCoded:
                continue

            # Compute time lag
            dt = frameIndicesPredicted - t1
            closest = np.argmin(np.abs(dt))
            t2 = frameIndicesPredicted[closest]
            lag = round(abs(((t1 - t2) / fps)), 3)

            #
            if (lag <= maximumLag) and saccadeLabelsPredicted[closest] == saccadeLabelCoded:
                matched.append(True)
            else:
                matched.append(False)
        matched = np.array(matched)
        tpr = round(matched.sum() / matched.size, 3)
        result['tpr'][saccadeLabel] = tpr

        # Compute the false positive rate
        unmatched = list()
        frameIndicesTrueExcludingNaS = np.copy(frameIndicesTrue)
        frameIndicesTrueExcludingNaS = np.delete(
            frameIndicesTrueExcludingNaS,
            np.where(np.isnan(frameIndicesTrue))[0]
        )
        saccadeLabelsTrueExcludingNaS = np.copy(saccadeLabelsTrue)
        saccadeLabelsTrueExcludingNaS = np.delete(
            saccadeLabelsTrueExcludingNaS,
            np.where(np.isnan(frameIndicesTrue))[0]
        )
        for saccadeIndex, t1 in enumerate(frameIndicesPredicted):

            # Skip non-target samples
            if saccadeLabelsPredicted[saccadeIndex] != saccadeLabelCoded:
                continue

            # Find the closest true saccade that isn't an NaS
            dt = frameIndicesTrueExcludingNaS - t1
            closest = np.argmin(np.abs(dt))
            t2 = frameIndicesTrueExcludingNaS[closest]
            lag = round(abs(((t1 - t2) / fps)), 3)

            #
            if (lag <= maximumLag) and saccadeLabelsTrueExcludingNaS[closest] == saccadeLabelCoded:
                unmatched.append(False)
            else:
                unmatched.append(True)

        unmatched = np.array(unmatched)
        fpr = round(unmatched.sum() / unmatched.size, 3)
        result['fpr'][saccadeLabel] = fpr

    return result

def _computeOutcomeRates2(
    frameIndicesTrue,
    frameIndicesPredicted,
    saccadeLabelsTrue,
    saccadeLabelsPredicted,
    maximumLag,
    fps,
    ):
    """
    """

    # Compute the true positive rate
    matched = list()
    for t1, l1 in zip(frameIndicesTrue, saccadeLabelsTrue):

        # Compute time lag
        dt = frameIndicesPredicted - t1
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesPredicted[closest]
        lag = round(abs(((t1 - t2) / fps)), 3)

        #
        l2 = saccadeLabelsPredicted[closest]
        if (lag <= maximumLag) and (l1 == l2):
            matched.append(True) # True positive
        else:
            matched.append(False) # False negatie
    matched = np.array(matched)
    tpr = round(matched.sum() / matched.size, 3)

    # Compute the false positive rate
    unmatched = list()
    for t1, l1 in zip(frameIndicesPredicted, saccadeLabelsPredicted):

        # Find the closest true saccade that isn't an NaS
        dt = frameIndicesTrue - t1
        closest = np.argmin(np.abs(dt))
        t2 = frameIndicesTrue[closest]
        lag = round(abs(((t1 - t2) / fps)), 3)

        #
        l2 = saccadeLabelsTrue[closest]
        if (lag <= maximumLag) and (l1 == l2):
            unmatched.append(False) # True negative
        else:
            unmatched.append(True) # False positive

    unmatched = np.array(unmatched)
    fpr = round(unmatched.sum() / unmatched.size, 3)

    return tpr, fpr

def quantifyPerformance(
    manualLabeling,
    modelPredictions,
    interframeIntervals,
    maximumLag=0.005,
    ):
    """
    """

    #
    fps = round(1 / (np.median(np.loadtxt(interframeIntervals)) / 1000000000), 3)

    # Extract frame indices and labels for manually labeled saccades
    with h5py.File(manualLabeling, 'r') as stream:
        onsetLags = np.array(stream['saccade_onset']).ravel()
        frameIndicesByWaveform = np.array(stream['frame_indices'])
        frameIndicesTrue = list()
        for saccadeIndex, i1 in enumerate(frameIndicesByWaveform):
            i2 = np.interp(0.5, [0, 1], [i1.min(), i1.max()]).item()
            i2 += round(onsetLags[saccadeIndex] * fps, 3)
            frameIndicesTrue.append(i2)
        frameIndicesTrue = np.array(frameIndicesTrue)
        saccadeLabelsTrue = np.array(stream['saccade_labels'])
        NaS = np.isnan(saccadeLabelsTrue)
        frameIndicesTrue = np.delete(frameIndicesTrue, np.where(NaS)[0])
        saccadeLabelsTrue = np.delete(saccadeLabelsTrue, np.where(NaS)[0])


    # Extract frame indices and labels for predicted saccades
    with h5py.File(modelPredictions, 'r') as stream:
        frameIndicesPredicted = np.array(stream['saccade_onset']).ravel()
        saccadeLabelsPredicted = np.array(stream['saccade_labels_coded'])

    #
    result = _computeOutcomeRates(
        frameIndicesTrue,
        frameIndicesPredicted,
        saccadeLabelsTrue,
        saccadeLabelsPredicted,
        maximumLag,
        fps
    )

    return result

def computeReceiverOperatingCurves(
    poseEstimates,
    manualLabeling,
    maximumLag=0.1,
    fps=30,
    ):
    """
    """

    # Compute projections
    pose, proj, uh, uv = computePoseProjections(
        poseEstimates
    )
    signal = np.diff(proj[:, 0])

    # Load manual labeling
    saccadeOnsetsTrue, saccadeLabelsDecoded = loadManualSaccadeLabeling(
        manualLabeling
    )
    saccadeLabelsTrue = list()
    for saccadeLabel in saccadeLabelsDecoded:
        if saccadeLabel == 'n':
            saccadeLabelsTrue.append(1)
        elif saccadeLabel == 't':
            saccadeLabelsTrue.append(-1)
        else:
            saccadeLabelsTrue.append(0)
    saccadeLabelsTrue = np.array(saccadeLabelsTrue)

    # Compute predicted saccade onsets and labels
    xy = list()
    thresholds = np.linspace(
        np.nanpercentile(np.abs(signal), 50),
        np.nanmax(np.abs(signal)),
        100
    )
    for threshold in thresholds:
        peakIndices, peakProps = find_peaks(
            np.abs(signal),
            height=0,
            distance=0.07 * fps
        )
        saccadeOnsetsPredicted = peakIndices
        saccadeLabelsPredicted = list()
        for peakIndex in peakIndices:
            if abs(signal[peakIndex]) < threshold:
                saccadeLabel = 0
            elif signal[peakIndex] < 0:
                saccadeLabel = 1
            else:
                saccadeLabel = -1
            saccadeLabelsPredicted.append(saccadeLabel)
        saccadeLabelsPredicted = np.array(saccadeLabelsPredicted)
        tpr, fpr = _computeOutcomeRates2(
            saccadeOnsetsTrue,
            saccadeOnsetsPredicted,
            saccadeLabelsTrue,
            saccadeLabelsPredicted,
            maximumLag,
            fps=fps
        )
        xy.append([tpr, fpr])

    return np.array(xy)

class SimpleThresholdingClassifier():
    """
    """

    def __init__(self, threshold):
        self.threshold = threshold
        return 
    
    
    def fit(self):
        return
    
    def predict(self, X):
        labels = list()
        for x in X:
            iMiddle = np.interp(0.5, [0, 1], [0, x.size]).item()
            yMiddle = round(np.interp(iMiddle, np.arange(x.size), x).item(), 3)
            if abs(yMiddle) < self.threshold:
                labels.append(0)
            elif yMiddle < 0:
                labels.append(1)
            elif yMiddle > 0:
                labels.append(-1)
            else:
                raise Exception()
        return np.array(labels)

from sklearn.metrics import confusion_matrix  
def computeReceiverOperatingCharacteristic(
    manualLabeling,
    ):
    """
    """

    with h5py.File(manualLabeling, 'r') as stream:
        saccadeWaveforms = np.array(stream['saccade_waveforms'])
        saccadeLabels = np.array(stream['saccade_labels'])
    
    #
    X = np.diff(saccadeWaveforms[:, 0, :], axis=1)
    yTrue = saccadeLabels

    #
    thresholds = np.linspace(
        np.percentile(np.abs(X.ravel()), 50),
        np.abs(X.ravel()).max(),
        100
    )
    xy = list()
    for threshold in thresholds:
        clf = SimpleThresholdingClassifier(threshold=threshold)
        yPred = clf.predict(X)
        cm = confusion_matrix(yTrue, yPred)

        # Compute per-class FPR and TPR
        num_classes = cm.shape[0]
        fpr_per_class = []
        tpr_per_class = []

        for i in range(num_classes):

            # Compute FPR
            FP = np.sum(cm[:, i]) - cm[i, i]  # False Positives for class i
            TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])  # True Negatives for class i
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
            fpr_per_class.append(fpr)

            # Compute TPR
            TP = cm[i, i]  # True Positives for class i
            FN = np.sum(cm[i, :]) - TP  # False Negatives for class i
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            tpr_per_class.append(tpr)

        #
        # fpr_macro = np.mean(fpr_per_class)
        # tpr_macro = np.mean(tpr_per_class)
        xy.append([fpr_per_class, tpr_per_class])
    xy = np.array(xy)

    fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True)
    for i in range(3):
        axs[0, i].plot(thresholds, xy[:, 1, i])
        axs[0, i].plot(thresholds, xy[:, 0, i])
        axs[1, i].plot(xy[:, 0, i], xy[:, 1, i])

    #
    optimums = list()
    for i in range(3):
        tpr = xy[:, 1, i]
        fpr = xy[:, 0, i]
        dists = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        index = np.argmin(dists)
        x = xy[index, 0, i]
        y = xy[index, 1, i]
        axs[1, i].scatter(x, y, color='r')
        optimums.append(thresholds[index])
    optimum = np.mean(optimums)
    for ax in axs[0, :]:
        ylim = ax.get_ylim()
        ax.vlines(optimum, *ylim, color='r')
        ax.set_ylim(ylim)

    #
    for i in range(3):
        tpr = np.interp(optimum, thresholds, xy[:, 1, i]).item()
        fpr = np.interp(optimum, thresholds, xy[:, 0, i]).item()
        print(f'Result for class {i + 1}: FPR={fpr:.2f}, TPR={tpr:.2f}')

    return fig, axs, optimum