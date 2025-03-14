import yaml
import polars
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from decimal import Decimal

def loadPoseEstimates(
    file
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

    return df

def projectPoseEstimates(
    frame,
    points,
    normalize=False # TODO: Make normalization optional
    ):
    """
    Project pupil center onto nasal-temporal and upper-lower axes
    
    Notes
    -----
    Axes are signed such that negative values mean nasal/upper whereas positive
    values mean temporal/lower (relative to the centroid)
    """

    # Unpack the dictionary
    nasal = frame['N']
    temporal = frame['T']
    lower = frame['L']
    upper = frame['U']

    # Vectors for the axes
    nt = np.array([temporal[0] - nasal[0], temporal[1] - nasal[1]])
    ul = np.array([lower[0] - upper[0], lower[1] - upper[1]])

    # Points to project
    x = points[:, 0]
    y = points[:, 1]

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

    return magnitude, uHorizontal, uVertical

def computePoseProjections(
    dlcFile,
    likelihoodThreshold=0.95
    ):
    """
    """

    # Load data
    pose = loadPoseEstimates(dlcFile)

    #
    frame = dict()
    for pt in ('N', 'L', 'T', 'U'):
        frame[pt] = np.array(pose.select(f'{pt}_x', f'{pt}_y')).mean(0)
    P = np.array(pose.select('P_x', 'P_y'))
    likelihood = np.array(pose['P_likelihood'])
    P[likelihood < likelihoodThreshold, :] = (np.nan, np.nan)
    projections, uHorizontal, uVertical = projectPoseEstimates(frame, P)


    return pose, projections, uHorizontal, uVertical

def extractPutativeSaccades(
    configFile,
    poseEstimates,
    interframeIntervals,
    likelihoodThreshold=0.95,
    maximumFrameLoss=0.15,
    maximumFrameDifference=0.01,
    ):
    """
    """

    #
    with open(configFile, 'r') as stream:
        configData = yaml.safe_load(stream)

    # Load pose and projections onto N-T and U-L axes
    pose, projections, uHorizontal, uVertical = computePoseProjections(poseEstimates, likelihoodThreshold)
    nFrames = pose.shape[0]

    # Check how much of the eye position data is NaN values
    frameLoss = np.isnan(projections[:, 0]).sum() / projections.shape[0]
    if frameLoss > maximumFrameLoss:
        raise Exception(f'{frameLoss * 100:.2f}% of pose estimates are NaN values (more than threshold of {maximumFrameLoss * 100:.0f}%)')

    # Compute the empirical framerate
    ifi = np.loadtxt(interframeIntervals)[1:] / 1000000000 # Drop the first interval
    fps = 1 / np.median(ifi)
    diff = ifi.size + 1 - nFrames # Difference in the number of frames
    if diff != 0:
        print(f'WARNING: The number of frames ({nFrames}) is different than the number of timestamps ({ifi.size + 1})')
        if diff / nFrames > maximumFrameDifference:
            raise Exception(f'Difference in frame count ({diff / nFrames:.2f}) is exceeds threshold ({maximumFrameDifference:.2f})')
        else:
            print(f'WARNING: Assuming a constant framerate of {fps:.2f} fps')
            ifi = np.full(nFrames - 1, np.median(ifi))

    #
    tFrames = np.concatenate([[0,], np.cumsum(ifi)])

    # Smooth signal
    sigma = round(fps * configData['smoothingWindowSize'], 2)
    horizontalEyePosition = gaussian_filter1d(projections[:, 0], sigma=sigma)
    horizontalEyeVelocity = np.diff(horizontalEyePosition)
    verticalEyePosition = gaussian_filter1d(projections[:, 1], sigma=sigma)

    # Detect peaks
    heightThreshold = np.nanpercentile(horizontalEyeVelocity, configData['velocityThreshold'])
    distanceThreshold = np.ceil(configData['minimumPeakDistance'] * fps)
    peakIndices, peakProps = find_peaks(
        np.abs(horizontalEyeVelocity),
        height=heightThreshold,
        distance=distanceThreshold
    )

    #
    saccadeLoss = 0
    frameIndices = list()
    frameTimestamps = list()
    saccadeWaveforms = list()

    for peakIndex in peakIndices:
        

        #
        tPeak = (tFrames[peakIndex] + tFrames[peakIndex + 1]) / 2
        tLeft = float(Decimal(str(tPeak)) + Decimal(str(configData['responseWindow'][0])))
        tRight = float(Decimal(str(tPeak)) + Decimal(str(configData['responseWindow'][1])))
        tEval = np.linspace(tLeft, tRight, configData['waveformSize'] + 1)

        #
        iEval = np.around(np.interp(
            tEval,
            tFrames,
            np.arange(tFrames.size)
        ), 3)

        #
        wf = np.array([
            np.interp(tEval, tFrames, horizontalEyePosition),
            np.interp(tEval, tFrames, verticalEyePosition)
        ])

        #
        if np.isnan(wf).sum() > 0:
            saccadeLoss += 1
            continue

        #
        saccadeWaveforms.append(wf)
        frameTimestamps.append(tEval)
        frameIndices.append(iEval)

    #
    frameIndices = np.around(np.array(frameIndices), 3)
    frameTimestamps = np.around(np.array(frameTimestamps), 3)
    saccadeWaveforms = np.around(np.array(saccadeWaveforms), 3)
    print(f'INFO: {saccadeLoss} out of {peakIndices.size} putative saccades lost due to uncertainty in pose estimation')

    return saccadeWaveforms, frameIndices, frameTimestamps