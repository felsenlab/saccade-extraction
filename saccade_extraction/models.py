import yaml
import pickle
import numpy as np
import pathlib as pl
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import r2_score, make_scorer
import h5py

class MLPRegressorWithStandardization(MLPRegressor):
    """
    """

    def __init__(self, mu=None, sigma=None, hidden_layer_sizes=(100,), activation='relu',
        solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', 
        learning_rate_init=0.001, max_iter=200, tol=1e-4, verbose=False
        ):
        """
        Custom MLPRegressor that includes standardization parameters (mu, sigma).
        """
        self.mu = mu
        self.sigma = sigma
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                        solver=solver, alpha=alpha, batch_size=batch_size,
                        learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                        max_iter=max_iter, tol=tol, verbose=verbose)

    def fit(self, X, y):
        """
        """

        self.mu = np.nanmean(y, axis=0)
        self.sigma = np.nanstd(y, axis=0)
        y = (y - self.mu) / self.sigma
        super().fit(X, y)

        return
    
    def predict(self, X):
        """
        """

        y = super().predict(X)

        return y * self.sigma + self.mu

def createTrainingDataset(
    configFile,
    tag=None
    ):
    """
    Create an aggregate training dataset for training models
    """

    with open(configFile, 'r') as stream:
        configData = yaml.safe_load(stream)
    projectDirectory = pl.Path(configData['projectDirectory'])
    X, y = list(), list()
    nSessions = 0
    for folder in projectDirectory.joinpath('data').iterdir():
        f = folder.joinpath('putative_saccades_data.hdf')
        if f.exists():
            with h5py.File(f, 'r') as stream:
                putativeSaccadeWaveforms = np.array(stream['saccade_waveforms'])[:, 0, :]
                putativeSaccadeLabels = np.array(stream['saccade_labels']).reshape(-1, 1)
                putativeSaccadeOnset = np.array(stream['saccade_onset']).reshape(-1, 1)
                putativeSaccadeOffset = np.array(stream['saccade_offset']).reshape(-1, 1)
            X_ = putativeSaccadeWaveforms
            y_ = np.concatenate([
                putativeSaccadeLabels,
                putativeSaccadeOnset,
                putativeSaccadeOffset
            ], axis=1)
        else:
            continue
        sampleIndices = np.where(np.logical_not(np.logical_or(
            np.isnan(X_).any(1),
            np.isnan(y_).all(1)
        )))[0]
        for sampleIndex in sampleIndices:
            X.append(X_[sampleIndex])
            y.append(y_[sampleIndex])
        nSessions += 1

    #
    X = np.array(X)
    y = np.array(y).reshape(-1, 3)
    nSamples = X.shape[0]
    print(f'Training dataset generated with {nSamples} samples from {nSessions} sessions')

    #
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if tag is None:
        folder = timestamp
    else:
        folder = f'{timestamp}_{tag}'
    targetDirectory = projectDirectory.joinpath('models', folder)
    if targetDirectory.exists() == False:
        targetDirectory.mkdir()
    np.save(targetDirectory.joinpath('samples.npy'), X)
    np.save(targetDirectory.joinpath('labels.npy'), y)

    return

def trainModels(
    configFile,
    trainingDataset=-1,
    layerSizeRange=(4, 20),
    ):
    """
    Train saccade direction classifier and epoch regressor
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
    targetDirectory = folders[sortedIndices[trainingDataset]]

    #
    print(f'Training models with training data from {targetDirectory.name} ...')

    # Load samples and labels
    X = np.load(targetDirectory.joinpath('samples.npy'))
    X = np.diff(X, axis=1) # Compute velocity
    X = X / np.abs(X).max(1).reshape(-1, 1) # Normalize to peak velocity
    y = np.load(targetDirectory.joinpath('labels.npy'))

    # Train clasifier
    inclusionMask = np.invert(np.isnan(y[:, 0]))
    X1, y1 = X[inclusionMask, :], y[inclusionMask, 0]
    hiddenLayerSizes = list()
    for i in range(3):
        for j in range(layerSizeRange[0], layerSizeRange[1] + 1, 1):
            hiddenLayerSizes.append(np.repeat(j, np.power(i, 2)))
    grid = {
        'hidden_layer_sizes': hiddenLayerSizes,
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    }
    clf_ = MLPClassifier(
        max_iter=10000,
        activation='relu',
        solver='adam',
        learning_rate='constant'
    )
    search = GridSearchCV(
        clf_,
        grid,
        cv=3,
    )
    search.fit(X1, y1.ravel())
    clf = search.best_estimator_

    # Report performance
    score = search.best_score_
    print(f"Best classifier selected, score={score:.2f}")

    # Fit regressor
    inclusionMask = np.invert(np.isnan(y[:, 1:]).any(1))
    X2, y2 = X[inclusionMask, :], y[inclusionMask, 1:]
    for i in range(3):
        for j in range(layerSizeRange[0], layerSizeRange[1] + 1, 1):
            hiddenLayerSizes.append(np.repeat(j, np.power(i, 2)))
    grid = {
        'hidden_layer_sizes': hiddenLayerSizes,
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'learning_rate_init': [0.01, 0.001, 0.0001]
    }
    reg_ = MLPRegressorWithStandardization(
        max_iter=10000,
        solver='adam',
        learning_rate='constant',
        activation='relu',
    )
    search = GridSearchCV(
        reg_,
        grid,
        cv=3,
        # scoring=make_scorer(r2_score)
    )
    search.fit(X2, y2)
    reg = search.best_estimator_
    score = search.best_score_
    print(f'Best regressor selected, score={score:.2f}')

    # Save trained models
    with open(targetDirectory.joinpath('classifier.pkl'), 'wb') as stream:
        pickle.dump(clf, stream)
    with open(targetDirectory.joinpath('regressor.pkl'), 'wb') as stream:
        pickle.dump(reg, stream)

    return