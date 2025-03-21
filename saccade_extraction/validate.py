from matplotlib import pylab as plt
import numpy as np
import h5py
import pathlib as pl
import shutil
from sklearn.metrics import confusion_matrix 
import pickle

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
 
def computeRocCurves(
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
        fprByClass = []
        tprByClass = []

        for i in range(cm.shape[0]):

            # Compute FPR
            fp = np.sum(cm[:, i]) - cm[i, i]  # False Positives for class i
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])  # True Negatives for class i
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprByClass.append(fpr)

            # Compute TPR
            tp = cm[i, i]  # True Positives for class i
            fn = np.sum(cm[i, :]) - tp  # False Negatives for class i
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tprByClass.append(tpr)

        #
        xy.append([fprByClass, tprByClass])

    #
    xy = np.array(xy)

    #
    fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True)
    for i in range(3):
        axs[0, i].plot(thresholds, xy[:, 1, i])
        axs[0, i].plot(thresholds, xy[:, 0, i])
        axs[1, i].plot(xy[:, 0, i], xy[:, 1, i])

    #
    opts = list()
    for i in range(3):
        tpr = xy[:, 1, i]
        fpr = xy[:, 0, i]
        dists = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        index = np.argmin(dists)
        x = xy[index, 0, i]
        y = xy[index, 1, i]
        axs[1, i].scatter(x, y, color='r')
        opts.append(thresholds[index])
    opt = np.mean(opt)

    #
    for ax in axs[0, :]:
        ylim = ax.get_ylim()
        ax.vlines(opt, *ylim, color='r')
        ax.set_ylim(ylim)

    #
    for i in range(3):
        tpr = np.interp(opt, thresholds, xy[:, 1, i]).item()
        fpr = np.interp(opt, thresholds, xy[:, 0, i]).item()
        print(f'Result for class {i + 1}: fpr={fpr:.2f}, tpr={tpr:.2f}')

    return fig, axs, opt

def quantifyPerformance(
    manualLabeling,
    clf,
    ):
    """
    """

    with h5py.File(manualLabeling, 'r') as stream:
        saccadeWaveforms = np.array(stream['saccade_waveforms'])
        saccadeLabels = np.array(stream['saccade_labels'])

    #
    if type(clf) == str:
        with open(clf, 'rb') as stream:
            clf = pickle.load(clf)
    
    #
    X = np.diff(saccadeWaveforms[:, 0, :], axis=1)
    yTrue = saccadeLabels
    yPred = clf.predict(X)
    cm = confusion_matrix(yTrue, yPred)

    # Compute per-class FPR and TPR
    fprByClass = []
    tprByClass = []

    for i in range(cm.shape[0]):

        # Compute FPR
        fp = np.sum(cm[:, i]) - cm[i, i]  # False Positives for class i
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])  # True Negatives for class i
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fprByClass.append(fpr)

        # Compute TPR
        tp = cm[i, i]  # True Positives for class i
        fn = np.sum(cm[i, :]) - tp  # False Negatives for class i
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tprByClass.append(tpr)

    return np.array(fprByClass), np.array(tprByClass)