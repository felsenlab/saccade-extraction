from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QSizePolicy, QFileDialog, QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import ticker
from matplotlib.pylab import subplots, barh
import numpy as np
import pathlib as pl
import sys
import yaml
import h5py

class UserInterface(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1200, 700)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setContentsMargins(-1, -1, 100, -1)
        self.mainLayout.setObjectName("mainLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton_4.sizePolicy().hasHeightForWidth())
        self.radioButton_4.setSizePolicy(sizePolicy)
        self.radioButton_4.setObjectName("radioButton_4")
        self.verticalLayout.addWidget(self.radioButton_4)
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton.sizePolicy().hasHeightForWidth())
        self.radioButton.setSizePolicy(sizePolicy)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton_2.sizePolicy().hasHeightForWidth())
        self.radioButton_2.setSizePolicy(sizePolicy)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton_3.sizePolicy().hasHeightForWidth())
        self.radioButton_3.setSizePolicy(sizePolicy)
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout.addWidget(self.radioButton_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setBaseSize(QtCore.QSize(0, 0))
        self.comboBox.setEditable(True)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_2.sizePolicy().hasHeightForWidth())
        self.checkBox_2.setSizePolicy(sizePolicy)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.checkBox)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.matplotlibWidget_2 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_2.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_2.setSizePolicy(sizePolicy)
        self.matplotlibWidget_2.setObjectName("matplotlibWidget_2")
        self.verticalLayout.addWidget(self.matplotlibWidget_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.mainLayout.addLayout(self.verticalLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.matplotlibWidget_1 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_1.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_1.setSizePolicy(sizePolicy)
        self.matplotlibWidget_1.setObjectName("matplotlibWidget_1")
        self.horizontalLayout_3.addWidget(self.matplotlibWidget_1)
        self.horizontalLayout_3.setStretch(0, 1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout.addWidget(self.pushButton_6)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout.addWidget(self.pushButton_5)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.mainLayout.addLayout(self.verticalLayout_4)
        self.mainLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.mainLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.comboBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Saccade Labeling GUI"))
        self.label.setText(_translate("MainWindow", "Labels:"))
        self.radioButton_4.setText(_translate("MainWindow", "Not labeled"))
        self.radioButton.setText(_translate("MainWindow", "Not a saccade"))
        self.radioButton_2.setText(_translate("MainWindow", "Nasal saccade"))
        self.radioButton_3.setText(_translate("MainWindow", "Temporal saccade"))
        self.label_4.setText(_translate("MainWindow", "Sample order:"))
        self.comboBox.setCurrentText(_translate("MainWindow", "Random"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Random"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Chronological"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Labeled"))
        self.label_2.setText(_translate("MainWindow", "Line selector:"))
        self.checkBox_2.setText(_translate("MainWindow", "Start"))
        self.checkBox.setText(_translate("MainWindow", "stop"))
        self.label_5.setText(_translate("MainWindow", "Metrics:"))
        self.pushButton_3.setText(_translate("MainWindow", "Open"))
        self.pushButton_6.setText(_translate("MainWindow", "Close"))
        self.pushButton_4.setText(_translate("MainWindow", "Save"))
        self.pushButton.setText(_translate("MainWindow", "Previous"))
        self.pushButton_2.setText(_translate("MainWindow", "Next"))
        self.pushButton_5.setText(_translate("MainWindow", "Reset"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))

class FigureCanvasExtended(FigureCanvas):
    """
    """

    def __init__(self, parent=None, gui=None,  nrows=1, ncols=1):
        self.fig, self.axes = subplots(nrows, ncols, sharex=True, sharey=True)
        self.parent = parent
        super().__init__(self.fig)
        self.gui = gui
        self.mpl_connect('button_press_event', self.onMouseClick)
        return
    
    def onMouseClick(self, event):
        """
        """

        if self.parent and self.parent.toolbar.mode != '':
            return
        self.gui.handleMouseClickForMatplotlib(event)

        return
    
class SaccadeLabelingWidget(QWidget):
    """
    """

    def __init__(self, parent=None, gui=None, nrows=1, ncols=1, tb=True):
        """
        """

        super().__init__(parent)
        self.canvas = FigureCanvasExtended(gui=gui, parent=self, nrows=nrows, ncols=ncols)

        # Add subplots
        self.axes = self.canvas.axes
        #
        self.axes[0, 0].set_title('Nasal-temporal component', fontsize=10)
        self.axes[0, 1].set_title('Upper-lower component', fontsize=10)
        self.axes[0, 0].set_ylabel('Nasal <- Position (px) -> Temp.')
        self.axes[1, 0].set_ylabel('Velocity (px/s)')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 1].set_xlabel('Time (s)')

        # Add toolbar
        if tb:
            self.toolbar = NavigationToolbar(self.canvas, self)
        else:
            self.toolbar = None

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)  # Add toolbar on top
        layout.addWidget(self.canvas)   # Add figure below it
        self.setLayout(layout)

        # Add lines
        self.lines = {
            'start': np.full([2, 2], object),
            'stop': np.full([2, 2], object),
            'vline': np.full([2, 2], object),
            'hline': np.full([2, 2], object),
            'wave': np.full([2, 2], object),
        }
        self.patches = np.full([2, 2], object)
        for (i, j), ax in np.ndenumerate(self.axes):

            # Lines
            for key in self.lines.keys():
                line = Line2D([], [], color='k', alpha=0.3)
                ax.add_line(line)
                line.set_visible(False)
                self.lines[key][i, j] = line

            # Patches
            patch = Rectangle([0, 0], 0, 0)
            ax.add_patch(patch)
            patch.set_facecolor('k')
            patch.set_alpha(0.1)
            patch.set_visible(False)
            self.patches[i, j] = patch

        return

class LabelingMetricsWidget(QWidget):
    """
    """

    def __init__(self, parent=None):
        """
        """

        super().__init__(parent)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = [self.canvas.figure.add_subplot(111)]
        self.axes[0].set_xlabel('Counts')
        self.axes[0].set_ylabel('Sample label')
        self.bars = self.axes[0].barh(
            ['X', 'N', 'T'],
            np.zeros(3),
            height=0.4,
            color=['0.5', 'r', 'b']
        )
        self.labels = list()
        for i in range(3):
            label = self.axes[0].text(
                0, 0,
                '',
                ha='center',
                va='center',
                fontsize=10,
                color='w'
            )
            self.labels.append(label)
            label.set_visible(False)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        return

class SaccadeLabelingGUI(QMainWindow):
    """
    """

    def __init__(
        self,
        configFile,
        seek=None,
        ):
        """
        """

        #
        super().__init__()
        self.ui = UserInterface()
        self.ui.setupUi(self)

        # Subplots widget
        self.labelingWidget = SaccadeLabelingWidget(gui=self, nrows=2, ncols=2)
        layout = self.ui.matplotlibWidget_1.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.matplotlibWidget_1)  # Assign layout to correct parent
            self.ui.matplotlibWidget_1.setLayout(layout)  # Ensure the widget gets a layout
        layout.addWidget(self.labelingWidget)
        self.labelingWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.labelingWidget.canvas.fig.set_tight_layout(True)
        self.labelingWidget.canvas.draw()

        # Metrics widget
        self.metricsWidget = LabelingMetricsWidget()
        self.metricsWidget.setMaximumWidth(200)
        self.metricsWidget.setMaximumHeight(300)
        layout = self.ui.matplotlibWidget_2.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.matplotlibWidget_2)
            self.ui.matplotlibWidget_2.setLayout(layout)
        layout.addWidget(self.metricsWidget)
        self.metricsWidget.canvas.figure.set_tight_layout(True)
        self.metricsWidget.canvas.draw()

        #
        self.update()
        self.repaint()

        #
        self.X = None
        self.y = None
        self.targetDirectory = None
        self.saccadeWaveforms = None
        self.frameIndices = None
        self.frameTimestamps = None
        if seek is None:
            self.sampleIndex = None
        else:
            self.sampleIndex = seek - 1
        with open(configFile, 'r') as stream:
            self.configData = yaml.safe_load(stream)
        self.tStart = self.configData['responseWindow'][0]
        self.tStop = self.configData['responseWindow'][1]
        self.store = None
        self._i = 0

        #
        self.ui.pushButton_3.clicked.connect(self.onOpenButtonClicked)
        self.ui.actionLoad.triggered.connect(self.onOpenButtonClicked)
        self.ui.pushButton_2.clicked.connect(self.onNextButtonClicked)
        self.ui.pushButton_2.setToolTip('Push to load the next saccade')
        self.ui.pushButton.clicked.connect(self.onPreviousButtonClicked)
        self.ui.pushButton.setToolTip('Push to load the previous saccade')
        self.ui.radioButton.toggled.connect(self.onRadioButtonClicked)
        self.ui.radioButton_2.toggled.connect(self.onRadioButtonClicked)
        self.ui.radioButton_3.toggled.connect(self.onRadioButtonClicked)
        self.ui.radioButton_4.toggled.connect(self.onRadioButtonClicked)
        self.ui.pushButton_4.clicked.connect(self.onSaveButtonClicked)
        self.checkBoxes = (
            self.ui.checkBox_2,
            self.ui.checkBox
        )
        for cb in self.checkBoxes:
            cb.setEnabled(False)
            cb.clicked.connect(self.onCheckBoxClick)
        self.ui.checkBox_2.setToolTip('Enable to move the saccade onset line (you can also press A)')
        self.ui.checkBox.setToolTip('Enable to move the saccade offset line (you can also press Z)')
        self.ui.pushButton_5.clicked.connect(self.onResetLinesButtonClicked)
        self.ui.pushButton_5.setToolTip('Push to reset the saccade onset and offset lines')
        self.ui.pushButton_6.clicked.connect(self.onCloseButtonClicked)
        self.ui.comboBox.addItem('Amplitude (Descending)')
        self.ui.comboBox.addItem('Amplitude (Ascending)')
        self.ui.comboBox.setEditable(False)
        self.ui.comboBox.activated.connect(self.onComboBoxActivated)
        
        #
        self.initializePlots()

        # Disable all widgets except the Open button
        for widget in self.ui.centralwidget.findChildren(QWidget):
            if widget in [self.ui.pushButton_3, self.ui.pushButton_6]:
                continue
            else:
                widget.setEnabled(False)

        #
        self.installEventFilter(self)

        return
    
    def onOpenButtonClicked(self):
        """
        Load putative saccade waveforms
        """

        # Try to open file
        self.targetDirectory = pl.Path(QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            str(pl.Path(self.configData['projectDirectory']).joinpath('data')),
            QFileDialog.ShowDirsOnly
        ))

        # Load putative saccade waveforms
        self.store = self.targetDirectory.joinpath('putative_saccades_data.hdf')
        if self.store.exists() == False:
            self.store = None
            return
        with h5py.File(self.store, 'r') as stream:
            self.saccadeWaveforms = np.array(stream['saccade_waveforms'])
            self.frameIndices = np.array(stream['frame_indices'])
            self.frameTimestamps = np.array(stream['frame_timestamps'])

        #
        for widget in self.ui.centralwidget.findChildren(QWidget):
            if widget.isEnabled() == False: 
                widget.setEnabled(True)

        # Load labeled data
        nSamples = self.saccadeWaveforms.shape[0]
        self.sampleOrder = np.arange(nSamples)
        np.random.shuffle(self.sampleOrder)
        y = np.full([nSamples, 3], np.nan)
        with h5py.File(self.store, 'r') as stream:
            y[:, 0] = np.array(stream['saccade_labels']).ravel()
            y[:, 1] = np.array(stream['saccade_onset']).ravel()
            y[:, 2] = np.array(stream['saccade_offset']).ravel()
        self.y = y

        # Set the sample index
        if self.sampleIndex is None:
            self.sampleIndex = self.sampleOrder[0]

        # Determine the x and y limits for the axes
        ymax = np.max(np.abs(self.saccadeWaveforms))
        self.ylim = np.array([-ymax, ymax])

        # Update widgets
        self.onComboBoxActivated(None)
        self.updatePlots()
        self.updateMetricsWidget()
        self.updateRadioButtons()
        self.statusBar().showMessage(
            f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} from {self.targetDirectory.name}'
        )

        return
    
    def onSaveButtonClicked(self):
        """
        Save labeled data
        """

        #
        if self.store is None:
            return

        keys = (
            'saccade_labels',
            'saccade_onset',
            'saccade_offset'
        )
        with h5py.File(self.store, 'a') as stream:
            for i, k in enumerate(keys):
                if k in stream.keys():
                    del stream[k]
                data = self.y[:, i].reshape(-1, 1)
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data=data,
                    dtype=data.dtype
                )

        return
    
    def onPreviousButtonClicked(self):
        """
        """

        #
        self._i = np.take(np.arange(self.y.shape[0]), self._i - 1, mode='wrap')
        self.sampleIndex = self.sampleOrder[self._i]
        # self.sampleIndex = np.take(self.sampleIndices, self.sampleIndex - 1, mode='wrap')
        self.statusBar().showMessage(f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} for {self.targetDirectory.name}')
        self.updatePlots(resetLimits=True)
        self.updateRadioButtons()
        for cb in self.checkBoxes:
            cb.setChecked(False)

        return
    
    def onNextButtonClicked(self):
        """
        """

        #
        self._i = np.take(np.arange(self.y.shape[0]), self._i + 1, mode='wrap')
        self.sampleIndex = self.sampleOrder[self._i]
        # self.sampleIndex = np.take(self.sampleIndices, self.sampleIndex + 1, mode='wrap')
        self.statusBar().showMessage(f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} for {self.targetDirectory.name}')
        self.updatePlots(resetLimits=True)
        self.updateRadioButtons()
        for cb in self.checkBoxes:
            cb.setChecked(False)

        return
    
    def onResetLinesButtonClicked(self):
        """
        """

        self.y[self.sampleIndex, 1:] = (np.nan, np.nan)
        self.updatePlots(resetLimits=True)

        return
    
    def onCloseButtonClicked(self):
        """
        """

        self.onSaveButtonClicked() # Save progress
        self.close()

        return
    
    def onRadioButtonClicked(self):
        """
        """

        # Identify the button
        sender = self.sender()

        # Ignore the deselection signal
        if sender.isChecked() == False:
            return
        
        # Button selected
        if sender.isChecked():
            text = sender.text()
            if text == 'Not labeled':
                label = np.nan
            elif text == 'Not a saccade':
                label = 0
            elif text == 'Nasal saccade':
                label = 1
            elif text == 'Temporal saccade':
                label = -1
        
        # Button de-selected
        else:
            label = np.nan

        #
        self.y[self.sampleIndex, 0] = label
        self.updateMetricsWidget()

        return
    
    def onCheckBoxClick(
        self,
        ):
        """
        """

        #
        sender = self.sender()

        #
        for cb in self.checkBoxes:
            if cb != sender:
                if cb.isChecked():
                    cb.setChecked(False)

        return

    def onComboBoxActivated(self, index):
        """
        """

        text = self.ui.comboBox.currentText()
        nSamples = self.saccadeWaveforms.shape[0]
        if text == 'Random':
            sampleOrder = np.arange(nSamples)
            np.random.shuffle(sampleOrder)
        elif text == 'Chronological':
            sampleOrder = np.arange(nSamples)
        elif text == 'Labeled':
            sampleOrder = [i for i in np.where(np.logical_not(np.isnan(self.y).all(1)))[0]]
            for sampleIndex in np.arange(nSamples):
                if sampleIndex in sampleOrder:
                    continue
                sampleOrder.append(sampleIndex)
            sampleOrder= np.array(sampleOrder)
        elif text.lower().startswith('amplitude'):
            amplitudes = list()
            for wf in self.saccadeWaveforms[:, 0, :]:
                dp = np.diff(wf)
                iPeak = np.argmax(np.abs(dp))
                amplitudes.append(dp[iPeak])
            if 'descending' in text.lower():
                sampleOrder = np.argsort(amplitudes)[::-1]
            elif 'ascending' in text.lower():
                sampleOrder = np.argsort(amplitudes)

        self.sampleOrder = sampleOrder
        self._i = 0
        self.sampleIndex = self.sampleOrder[0]
        self.updatePlots()
        self.updateRadioButtons()
        self.statusBar().showMessage(
            f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} from {self.targetDirectory.name}'
        )

        return
    
    def eventFilter(self, obj, event):
        """
        """

        if event.type() == event.KeyPress:
            if event.key() == QtCore.Qt.Key_Left:
                self.onPreviousButtonClicked()
            elif event.key() == QtCore.Qt.Key_Right:
                self.onNextButtonClicked()
            elif event.key() == QtCore.Qt.Key_A:
                cb = self.checkBoxes[0]
                if cb.isChecked():
                    cb.setChecked(False)
                else:
                    cb.setChecked(True)
                    self.checkBoxes[1].setChecked(False)
            elif event.key() == QtCore.Qt.Key_Z:
                cb = self.checkBoxes[1]
                if cb.isChecked():
                    cb.setChecked(False)
                else:
                    cb.setChecked(True)
                    self.checkBoxes[0].setChecked(False)

        return super().eventFilter(obj, event)
    
    def handleMouseClickForMatplotlib(
        self,
        event
        ):
        """
        """

        if event.xdata is None:
            return
        x = round(event.xdata, 4)
        
        if self.checkBoxes[0].isChecked():
            if x < self.configData['responseWindow'][0]:
                x = np.nan
            self.y[self.sampleIndex, 1] = x
        elif self.checkBoxes[1].isChecked():
            if x > self.configData['responseWindow'][1]:
                x = np.nan
            self.y[self.sampleIndex, 2] = x
        self.updatePlots(resetLimits=False)

        return
    
    def updateRadioButtons(self):
        """
        Update the selection of the radio buttons
        """

        # Disconnect the radio buttons
        buttons = (
            self.ui.radioButton,
            self.ui.radioButton_2,
            self.ui.radioButton_3,
            self.ui.radioButton_4
        )
        for button in buttons:
            button.blockSignals(True)
        
        # Deselect all buttons
        for button in buttons:
            button.setChecked(False)

        # Select the appropriate button
        label = self.y[self.sampleIndex, 0]
        if np.isnan(label):
            button = self.ui.radioButton_4
        elif label == 0:
            button = self.ui.radioButton    
        elif label == 1:
            button = self.ui.radioButton_2
        elif label == -1:
            button = self.ui.radioButton_3
        button.setChecked(True)

        # Reconnect the radio buttons
        for button in buttons:
            button.blockSignals(False)

        return
    
    def initializePlots(self):
        """
        Initialize the matplotlib figures
        """

        # Set xticks and xlim
        self.xticks = np.around(np.arange(
            self.configData['responseWindow'][0],
            self.configData['responseWindow'][1] + 0.05,
            0.05,
        ), 2)
        self.xlim = (
            self.configData['responseWindow'][0] * 1.1,
            self.configData['responseWindow'][1] * 1.1
        )
        for ax in self.labelingWidget.axes.ravel():
            ax.set_xticks(self.xticks)
            ax.set_xticklabels(self.xticks)
            ax.set_xlim(self.xlim)

        return
    
    def updatePlots(self, resetLimits=True):
        """
        Update the matplotlib figures
        """

        # Compute timestamps
        t1 = np.linspace(
            self.configData['responseWindow'][0],
            self.configData['responseWindow'][1],
            self.configData['waveformSize'] + 1
        )
        t2 = np.interp(
            np.arange(0, t1.size - 1, 1) + 0.5,
            np.arange(t1.size),
            t1
        )

        # Update top-left plot
        ax = self.labelingWidget.axes[0, 0]

        # Draw position waveforms
        for j in (0, 1):
            self.labelingWidget.lines['wave'][0, j].set_xdata(t1)
            self.labelingWidget.lines['wave'][0, j].set_ydata(self.saccadeWaveforms[self.sampleIndex, j, :])
            self.labelingWidget.lines['wave'][0, j].set_alpha(1.0)
            if self.labelingWidget.lines['wave'][0, j].get_visible() == False:
                self.labelingWidget.lines['wave'][0, j].set_visible(True)

        # Draw velocity waveforms
        for j in (0, 1):
            self.labelingWidget.lines['wave'][1, j].set_xdata(t2)
            self.labelingWidget.lines['wave'][1, j].set_ydata(np.diff(self.saccadeWaveforms[self.sampleIndex, j, :]))
            self.labelingWidget.lines['wave'][1, j].set_alpha(1.0)
            if self.labelingWidget.lines['wave'][1, j].get_visible() == False:
                self.labelingWidget.lines['wave'][1, j].set_visible(True)    

        # Draw horizontal and vertical lines
        for (i, j), ax in np.ndenumerate(self.labelingWidget.axes):
            ln = self.labelingWidget.lines['hline'][i, j]
            ln.set_xdata(self.xlim)
            ln.set_ydata([0, 0])
            ln.set_visible(True)
            ln.set_linestyle(':')
            ln = self.labelingWidget.lines['vline'][i, j]
            ln.set_xdata([0, 0])
            ln.set_ydata(self.ylim)
            ln.set_visible(True)
            ln.set_linestyle(':')

        # Draw the start and stop lines
        for (i, j), ax in np.ndenumerate(self.labelingWidget.axes):

            #
            tStart = self.y[self.sampleIndex, 1]
            tStop = self.y[self.sampleIndex, 2]

            # Check if the sliders are set at the limits of the epoch range
            if tStart == self.configData['responseWindow'][0] or np.isnan(tStart):
                tStart = self.configData['responseWindow'][0]
                cStart = 'k'
                aStart = 0.3
            else:
                cStart = 'g'
                aStart = 0.7
            if tStop >= self.configData['responseWindow'][1] or np.isnan(tStop):
                tStop = self.configData['responseWindow'][1]
                cStop = 'k'
                aStop = 0.3
            else:
                cStop = 'r'
                aStop = 0.7

            # Set x and y data for the start line
            ydata = self.ylim
            xdata = (tStart, tStart)
            if np.isnan(xdata).all():
                xdata = (self.configData['responseWindow'][0], self.configData['responseWindow'][0])
            self.labelingWidget.lines['start'][i, j].set_xdata(xdata)
            self.labelingWidget.lines['start'][i, j].set_ydata(ydata)

            # Set x and y data for the stop line
            xdata = (tStop, tStop)
            if np.isnan(xdata).all():
                xdata = (self.configData['responseWindow'][1], self.configData['responseWindow'][1])
            self.labelingWidget.lines['stop'][i, j].set_xdata(xdata)
            self.labelingWidget.lines['stop'][i, j].set_ydata(ydata)

            # Set color and alpha for both lines
            lines = [
                self.labelingWidget.lines['start'][i, j],
                self.labelingWidget.lines['stop'][i, j]
            ]
            alphas = [aStart, aStop]
            colors = [cStart, cStop]
            for ln, a, c in zip(lines, alphas, colors):
                ln.set_visible(True)
                ln.set_color(c)
                ln.set_alpha(a)

        # Draw the patches
        for (i, j), ax in np.ndenumerate(self.labelingWidget.axes):
            patch = self.labelingWidget.patches[i, j]
            patch.set_xy([
                -1 * self.configData['minimumPeakDistance'],
                self.ylim[0]
            ])
            patch.set_width(2 * self.configData['minimumPeakDistance'])
            patch.set_height(np.diff(self.ylim).item())
            if patch.get_visible() == False:
                patch.set_visible(True)

        # Set the x and y limits
        if resetLimits:
            for ax in self.labelingWidget.axes.ravel():
                ax.set_ylim(self.ylim)
                ax.set_xlim(self.xlim)


        # Draw
        self.labelingWidget.canvas.draw()

        return

    def updateMetricsWidget(self):
        """
        """

        # Update the metrics widget
        counts = np.array([
            np.sum(self.y[:, 0] ==  0),
            np.sum(self.y[:, 0] == +1),
            np.sum(self.y[:, 0] == -1)
        ])
        for i in range(3):
            bar = self.metricsWidget.bars[i]
            bar.set_width(counts[i])
            label = self.metricsWidget.labels[i]
            label.set_position([
                bar.get_width() / 2,
                i
            ])
            label.set_text(f'{counts[i]:.0f}')
            label.set_visible(True)
        if np.max(counts) != 0:
            self.metricsWidget.axes[0].set_xlim([0, 1.1 * np.max(counts)])
        self.metricsWidget.canvas.draw()

        return
    
def launchGUI(configFile, sampleIndex=None):
    """
    """

    app = QtWidgets.QApplication.instance()  # Check if QApplication already exists
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    gui = SaccadeLabelingGUI(configFile, sampleIndex)
    gui.show()
    app.exec_()
    app.quit()

    return gui

if __name__ == "__main__":
    launchGUI()