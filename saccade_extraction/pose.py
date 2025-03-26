import yaml
import polars
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from decimal import Decimal

def _loadPoseEstimates(
    file,
    likelihoodThreshold=0.8,
    maximumDataLoss=0.1,
    ):
    """
    """

    headerLines = list()
    data = list()
    with open(file, 'r') as stream:
        for i, ln in enumerate(stream):
            if i < 3:
                headerLines.append(ln)
                continue
            elements = ln.strip('\r').split(',')[1:]
            data.append([round(float(el), 4) for el in elements])

    #
    bodypartLabels = headerLines[1].strip('\n').split(',')[1:]
    bodypartFeatures = headerLines[2].strip('\n').split(',')[1:]
    columnLabels = list()
    for l, f in zip(bodypartLabels, bodypartFeatures):
        columnLabels.append(f'{l}_{f}')

    df = polars.DataFrame(
        data,
        schema=columnLabels,
        orient='row',
    )

    pose = dict()
    for pt in ('eye_nasal', 'eye_ventral', 'eye_temporal', 'eye_dorsal', 'pupil_center'):
        pose[pt] = np.array(df.select(f'{pt}_x', f'{pt}_y'))
        l = np.array(df.select(f'{pt}_likelihood')).ravel()
        pose[pt][l < likelihoodThreshold, :] = np.array([np.nan, np.nan])

    #
    loss = np.isnan(pose['pupil_center']).all(1).sum() / pose['pupil_center'].shape[0]
    if loss > maximumDataLoss:
        raise Exception(f'Data loss due to uncertainty in pose estimation ({loss * 100:.1f}%) exceeds threshold of {maximumDataLoss * 100:.1f}')

    return pose

def _computeProjections(
    pose,
    normalize=False # TODO: Make normalization optional
    ):
    """
    Project pupil center onto nasal-temporal and upper-lower axes
    
    Notes
    -----
    Axes are signed such that negative values mean nasal/upper whereas positive
    values mean temporal/lower (relative to the centroid)
    """

    nasal = np.nanmean(pose['eye_nasal'], axis=0)
    temporal = np.nanmean(pose['eye_temporal'], axis=0)
    lower = np.nanmean(pose['eye_ventral'], axis=0)
    upper = np.nanmean(pose['eye_dorsal'], axis=0)

    # Vectors for the axes
    nt = np.array([temporal[0] - nasal[0], temporal[1] - nasal[1]])
    ul = np.array([lower[0] - upper[0], lower[1] - upper[1]])

    # Points to project
    x = pose['pupil_center'][:, 0]
    y = pose['pupil_center'][:, 1]

    # Projection onto the nasal-temporal axis
    magnitude = list()
    w = np.array([x - nasal[0], y - nasal[1]]).T
    projection = np.dot(w, nt) / np.dot(nt, nt)
    uHorizontal = np.vstack([
        projection * nt[0],
        projection * nt[1]
    ]).T
    magnitude.append(np.linalg.norm(uHorizontal, axis=1))

    # Projeciton onto the upper-lower axis
    w = np.array([x - upper[0], y - upper[1]]).T
    projection = np.dot(w, ul) / np.dot(ul, ul)
    uVertical = np.vstack([
        projection * ul[0],
        projection * ul[1]
    ]).T
    magnitude.append(np.linalg.norm(uVertical, axis=1))
    magnitude = np.array(magnitude).T
    magnitude[:, 0] -= np.nanmean(magnitude[:, 0])
    magnitude[:, 1] -= np.nanmean(magnitude[:, 1])

    return magnitude

def _identifyDroppedFrames(
    projections,
    interframeIntervals
    ):
    """
    """

    # Quality check
    ifi = np.loadtxt(interframeIntervals)[1:]
    n = projections.shape[0]
    if ifi.size + 1 != n:
        raise Exception('Different number of frames and timestamps')

    #
    factor = np.median(ifi)
    arrayIndices = list()
    frameTimestamps = np.concatenate([
        [0,],
        np.cumsum(ifi)
    ])
    for i, frameInterval in enumerate(ifi):
        n = round(frameInterval / factor)
        if (n - 1) > 0:
            for j in range(n - 1):
                arrayIndices.append(i)
    arrayIndices = np.array(arrayIndices)
    if len(arrayIndices) != 0:
        corrected = np.insert(
            projections,
            arrayIndices,
            np.array([np.nan, np.nan]),
            axis=0
        )
        frameTimestamps = np.insert(
            frameTimestamps,
            arrayIndices + 1,
            np.nan
        )
    else:
        corrected = projections
    isnan = np.isnan(frameTimestamps)
    frameTimestamps[isnan] = np.interp(
        np.where(isnan)[0],
        np.arange(frameTimestamps.size)[np.logical_not(isnan)],
        frameTimestamps[np.logical_not(isnan)],
    )
    frameTimestamps = np.around(frameTimestamps / 1000000000, 6)

    return corrected, frameTimestamps

def _processProjections(
    projections,
    interframeIntervals,
    maximumGapSize=0.01,
    smoothingWindowSize=0.003,
    ):
    """
    """

    # Interpolate over missing data
    ifi = np.loadtxt(interframeIntervals)[1:]
    fps = 1 / (np.median(ifi) / 1000000000)
    interpolated = np.copy(projections)
    for j in range(projections.shape[1]):
        indices = list()
        startIndices = np.where(np.diff(np.isnan(projections[:, j])))[0][::2] + 1
        for startIndex in startIndices:
            stopIndex = None
            for i in range(startIndex, projections.shape[0]):
                if np.isnan(projections[i, j]).item() == True:
                    continue
                stopIndex = j
                break
            if stopIndex is None:
                continue
            dt = (stopIndex - startIndex) / fps
            if dt < maximumGapSize:
                for i in range(startIndex, stopIndex + 1, 1):
                    indices.append(i)
        if len(indices) == 0:
            continue
        interpolated = np.interp(
            np.arange(projections.shape[0]),
            indices,
            projections[:, j]
        )

    # Smooth signal
    sigma = round(fps * smoothingWindowSize, 2)
    smoothed = gaussian_filter1d(interpolated, sigma=sigma, axis=0)

    return smoothed

def loadEyePosition(
    poseEstimates,
    interframeIntervals,
    likelihoodThreshold=0.8,
    maximumGapSize=0.01,
    smoothingWindowSize=0.003,
    maximumDataLoss=0.1,
    ):
    """
    """

    pose = _loadPoseEstimates(
        poseEstimates,
        likelihoodThreshold,
        maximumDataLoss
    )
    projections = _computeProjections(pose)
    corrected, frameTimestamps = _identifyDroppedFrames(
        projections,
        interframeIntervals
    )
    processed = _processProjections(
        corrected,
        interframeIntervals,
        maximumGapSize,
        smoothingWindowSize
    )

    return processed, frameTimestamps

def extractPutativeSaccades(
    configFile,
    poseEstimates,
    interframeIntervals,
    likelihoodThreshold=0.8,
    maximumGapSize=0.01,
    maximumDataLoss=0.1,
    ):
    """
    """

    #
    with open(configFile, 'r') as stream:
        configData = yaml.safe_load(stream)

    #
    eyePosition, frameTimestamps = loadEyePosition(
        poseEstimates,
        interframeIntervals,
        likelihoodThreshold,
        maximumGapSize,
        configData['smoothingWindowSize'],
        maximumDataLoss
    )
    horizontalEyePosition = eyePosition[:, 0]
    verticalEyePosition = eyePosition[:, 1]
    horizontalEyeVelocity = np.diff(horizontalEyePosition)

    # Detect peaks
    fps = 1 / (np.median(np.loadtxt(interframeIntervals)) / 1000000000)
    heightThreshold = np.nanpercentile(horizontalEyeVelocity, configData['velocityThreshold'])
    distanceThreshold = np.ceil(configData['minimumPeakDistance'] * fps)
    peakIndices, peakProps = find_peaks(
        np.abs(horizontalEyeVelocity),
        height=heightThreshold,
        distance=distanceThreshold
    )

    #
    # saccadeLoss = 0
    evaluationIndices = list()
    evaluationTimestamps = list()
    saccadeWaveforms = list()
    for peakIndex in peakIndices:
        

        #
        tPeak = np.mean([frameTimestamps[peakIndex], frameTimestamps[peakIndex + 1]])
        tLeft = float(Decimal(str(tPeak)) + Decimal(str(configData['responseWindow'][0])))
        tRight = float(Decimal(str(tPeak)) + Decimal(str(configData['responseWindow'][1])))
        tEval = np.linspace(tLeft, tRight, configData['waveformSize'] + 1)

        #
        iEval = np.around(np.interp(
            tEval,
            frameTimestamps,
            np.arange(frameTimestamps.size)
        ), 3)

        #
        wf = np.array([
            np.interp(tEval, frameTimestamps, horizontalEyePosition),
            np.interp(tEval, frameTimestamps, verticalEyePosition)
        ])

        #
        if np.isnan(wf).sum() > 0:
            # saccadeLoss += 1
            continue

        #
        saccadeWaveforms.append(wf)
        evaluationTimestamps.append(tEval)
        evaluationIndices.append(iEval)

    #
    evaluationIndices = np.around(np.array(evaluationIndices), 3)
    evaluationTimestamps = np.around(np.array(evaluationTimestamps), 3)
    saccadeWaveforms = np.around(np.array(saccadeWaveforms), 3)
    # print(f'INFO: {saccadeLoss} out of {peakIndices.size} putative saccades lost due to uncertainty in pose estimation')

    return saccadeWaveforms, evaluationIndices, evaluationTimestamps