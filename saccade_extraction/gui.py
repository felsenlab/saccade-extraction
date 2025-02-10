from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QSizePolicy, QFileDialog, QWidget, QButtonGroup
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
from matplotlib import ticker
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
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.radioButton_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_4.setObjectName("radioButton_4")
        self.verticalLayout.addWidget(self.radioButton_4)
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout.addWidget(self.radioButton_3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.matplotlibWidget_1 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_1.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_1.setSizePolicy(sizePolicy)
        self.matplotlibWidget_1.setAutoFillBackground(False)
        self.matplotlibWidget_1.setObjectName("matplotlibWidget_1")
        self.verticalLayout_2.addWidget(self.matplotlibWidget_1)
        self.matplotlibWidget_2 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_2.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_2.setSizePolicy(sizePolicy)
        self.matplotlibWidget_2.setObjectName("matplotlibWidget_2")
        self.verticalLayout_2.addWidget(self.matplotlibWidget_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_3.addWidget(self.label_8)
        self.matplotlibWidget_3 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_3.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_3.setSizePolicy(sizePolicy)
        self.matplotlibWidget_3.setObjectName("matplotlibWidget_3")
        self.verticalLayout_3.addWidget(self.matplotlibWidget_3)
        self.matplotlibWidget_4 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.matplotlibWidget_4.sizePolicy().hasHeightForWidth())
        self.matplotlibWidget_4.setSizePolicy(sizePolicy)
        self.matplotlibWidget_4.setObjectName("matplotlibWidget_4")
        self.verticalLayout_3.addWidget(self.matplotlibWidget_4)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
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
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.checkBox)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalLayout.addWidget(self.horizontalSlider_3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
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
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Saccade Labeling GUI"))
        self.label.setText(_translate("MainWindow", "Labels"))
        self.radioButton_4.setText(_translate("MainWindow", "Not labeled"))
        self.radioButton.setText(_translate("MainWindow", "Not a saccade"))
        self.radioButton_2.setText(_translate("MainWindow", "Nasal saccade"))
        self.radioButton_3.setText(_translate("MainWindow", "Temporal saccade"))
        self.label_7.setText(_translate("MainWindow", "Nasal-temporal component"))
        self.label_8.setText(_translate("MainWindow", "Upper-lower component"))
        self.pushButton_3.setText(_translate("MainWindow", "Open"))
        self.pushButton_6.setText(_translate("MainWindow", "Close"))
        self.pushButton_4.setText(_translate("MainWindow", "Save"))
        self.pushButton.setText(_translate("MainWindow", "Previous"))
        self.pushButton_2.setText(_translate("MainWindow", "Next"))
        self.pushButton_5.setText(_translate("MainWindow", "Reset lines"))
        self.label_2.setText(_translate("MainWindow", "Line selector:"))
        self.checkBox_2.setText(_translate("MainWindow", "Start"))
        self.checkBox.setText(_translate("MainWindow", "stop"))
        self.label_4.setText(_translate("MainWindow", "y-range:"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))

class MatplotlibCanvasWidget(FigureCanvas):
    """ Matplotlib Figure as a QWidget """
    def __init__(self, parent=None, gui=None):
        self.gui = gui
        self.fig = Figure()
        self.ax = self.fig.add_subplot()
        self.start = Line2D([], [], color='k', alpha=0.3)
        self.stop = Line2D([], [], color='k', alpha=0.3)
        self.wf = Line2D([], [], color='k')
        self.vline = Line2D([], [], color='k', alpha=0.3)
        self.hline = Line2D([], [], color='k', alpha=0.3)
        self.lines = (
            self.start,
            self.stop,
            self.wf,
            self.vline,
            self.hline
        )
        for ln in self.lines:
            self.ax.add_line(ln)
            ln.set_visible(False)
        # self.ax.tick_params(axis='y', labelrotation=90)
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        self.fig.tight_layout()
        super().__init__(self.fig)
        self.mpl_connect('button_press_event', self.onMouseClick)
        return
    
    def onMouseClick(self, event):
        """
        """

        self.gui.handleMouseClickForMatplotlib(event)

        return

class SaccadeLabelingGUI(QMainWindow):
    """
    """

    def __init__(
        self,
        configFile,
        seek=None,
        yscale=1.0
        ):
        """
        """

        #
        super().__init__()
        self.ui = UserInterface()
        self.ui.setupUi(self)

        #
        self.axes = np.array([
            [MatplotlibCanvasWidget(gui=self), MatplotlibCanvasWidget(gui=self)],
            [MatplotlibCanvasWidget(gui=self), MatplotlibCanvasWidget(gui=self)],
        ])
        for widget in self.axes.ravel():
            widget.ax.set_xlabel('Time (s)')
        self.axes[0, 0].ax.set_ylabel('Nasal <- Pos (px) -> Temp')
        self.axes[0, 1].ax.set_ylabel('Upper <- Pos (px) -> Lower')
        for widget in self.axes[1, :]:
            widget.ax.set_ylabel(f'Velocity ($\Delta px$)')
        indices = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        widgets = (
            self.ui.matplotlibWidget_1,
            self.ui.matplotlibWidget_2,
            self.ui.matplotlibWidget_3,
            self.ui.matplotlibWidget_4
        )

        for (i, j), widget in zip(indices, widgets):
            if widget.layout() is None:  # Ensure it has a layout
                layout = QVBoxLayout(widget)
                widget.setLayout(layout)
            else:
                layout = widget.layout()
            layout.addWidget(self.axes[i, j])

            # Set expanding size policy
            self.axes[i, j].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Adjust layout inside figure
            self.axes[i, j].fig.set_tight_layout(True)
            self.axes[i, j].draw()

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
        self.yscale = yscale
        self.tStart = self.configData['responseWindow'][0]
        self.tStop = self.configData['responseWindow'][1]
        self.store = None

        #
        self.ui.pushButton_3.clicked.connect(self.onOpenButtonClicked)
        self.ui.actionLoad.triggered.connect(self.onOpenButtonClicked)
        self.ui.pushButton_2.clicked.connect(self.onNextButtonClicked)
        self.ui.pushButton_2.setToolTip('Push to load the next saccade')
        self.ui.pushButton.clicked.connect(self.onPreviousButtonClicked)
        self.ui.pushButton.setToolTip('Push to load the previous saccade')
        self.ui.horizontalSlider_3.setMinimum(1)
        self.ui.horizontalSlider_3.setMaximum(100)
        self.ui.horizontalSlider_3.setValue(int(round(self.yscale * 100)))
        self.ui.horizontalSlider_3.valueChanged.connect(self.onRangeSliderMovement)
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
        for widget in self.axes.flatten():
            widget.setToolTip('Click anywhere on the figure to move the saccade onset or offset lines.\nYou must have at least one line selector enabled.')

        #
        self.initializePlots()

        # Disable all widgets except the Open button
        for widget in self.ui.centralwidget.findChildren(QWidget):
            if widget == self.ui.pushButton_3:
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
        y = np.full([nSamples, 3], np.nan)
        with h5py.File(self.store, 'r') as stream:
            y[:, 0] = np.array(stream['saccade_labels']).ravel()
            y[:, 1] = np.array(stream['saccade_onset']).ravel()
            y[:, 2] = np.array(stream['saccade_offset']).ravel()
        self.y = y

        # Set the sample index
        if self.sampleIndex is None:
            self.sampleIndex = 0

        # Determine the x and y limits for the axes
        y1 = np.abs([
            self.saccadeWaveforms.min(),
            self.saccadeWaveforms.max()
        ]).max() * 1.1
        y2 = np.abs([
            np.diff(self.saccadeWaveforms, axis=-1).min(),
            np.diff(self.saccadeWaveforms, axis=-1).max()
        ]).max() * 1.1
        self.ylims = np.array([
            [y1, y1],
            [y2, y2]
        ])

        # Update widgets
        self.onRangeSliderMovement(self.yscale * 100)
        self.updatePlots()
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
        self.sampleIndex = np.take(np.arange(len(self.y)), self.sampleIndex - 1, mode='wrap')
        self.statusBar().showMessage(f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} for {self.targetDirectory.name}')
        self.updatePlots()
        self.updateRadioButtons()

        return
    
    def onNextButtonClicked(self):
        """
        """

        #
        self.sampleIndex = np.take(np.arange(len(self.y)), self.sampleIndex + 1, mode='wrap')
        self.statusBar().showMessage(f'Showing sample {self.sampleIndex + 1} out of {self.saccadeWaveforms.shape[0]} for {self.targetDirectory.name}')
        self.updatePlots()
        self.updateRadioButtons()

        return
    
    def onResetLinesButtonClicked(self):
        """
        """

        self.y[self.sampleIndex, 1:] = (np.nan, np.nan)
        self.updatePlots()

        return
    
    def onCloseButtonClicked(self):
        """
        """

        self.onSaveButtonClicked() # Save progress
        self.close()

        return
    
    def onRangeSliderMovement(self, value):
        """
        """

        normalized = value / 100
        scaled = 0.1 * (100 / 0.1) ** normalized
        self.yscale = scaled / 100
        self.updatePlots()

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
    
    def eventFilter(self, obj, event):
        """
        """

        if event.type() == event.KeyPress:
            if event.key() == QtCore.Qt.Key_A:
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
        self.updatePlots()

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
        for widget in self.axes.ravel():
            widget.ax.set_xticks(self.xticks)
            widget.ax.set_xticklabels(self.xticks)
            widget.ax.set_xlim(self.xlim)
        
        # Remove the yticklabels
        # for widget in self.axes.ravel():
        #     widget.ax.set_yticklabels([])

        return
    
    def updatePlots(self):
        """
        Update the matplotlib figures
        """

        # Compute timestamps
        t1 = np.linspace(
            self.configData['responseWindow'][0],
            self.configData['responseWindow'][1],
            self.configData['nFeatures'] + 1
        )
        t2 = np.interp(
            np.arange(0, t1.size - 1, 1) + 0.5,
            np.arange(t1.size),
            t1
        )

        # Draw position waveforms
        self.axes[0, 0].wf.set_xdata(t1)
        self.axes[0, 0].wf.set_ydata(self.saccadeWaveforms[self.sampleIndex, 0, :])
        if self.axes[0, 0].wf.get_visible() == False:
            self.axes[0, 0].wf.set_visible(True)
        self.axes[0, 1].wf.set_xdata(t1)
        self.axes[0, 1].wf.set_ydata(self.saccadeWaveforms[self.sampleIndex, 1, :])
        if self.axes[0, 1].wf.get_visible() == False:
            self.axes[0, 1].wf.set_visible(True)    

        # Draw velocity waveforms
        self.axes[1, 0].wf.set_xdata(t2)
        self.axes[1, 0].wf.set_ydata(np.diff(self.saccadeWaveforms[self.sampleIndex, 0, :]))
        if self.axes[1, 0].wf.get_visible() == False:
            self.axes[1, 0].wf.set_visible(True)    
        self.axes[1, 1].wf.set_xdata(t2)
        self.axes[1, 1].wf.set_ydata(np.diff(self.saccadeWaveforms[self.sampleIndex, 1, :]))
        if self.axes[1, 1].wf.get_visible() == False:
            self.axes[1, 1].wf.set_visible(True)

        # Draw horizontal and vertical lines
        for widget in self.axes[0, :]:
            widget.hline.set_xdata(self.xlim)
            widget.hline.set_ydata([0, 0])
            widget.vline.set_xdata([0, 0])
            widget.vline.set_ydata([
                self.ylims[0, 0] * self.yscale * -1,
                self.ylims[0, 0] * self.yscale
            ])
            widget.vline.set_visible(True)
            widget.hline.set_visible(True)
        for widget in self.axes[1, :]:
            widget.hline.set_xdata(self.xlim)
            widget.hline.set_ydata([0, 0])
            widget.vline.set_xdata([0, 0])
            widget.vline.set_ydata([
                self.ylims[1, 0] * self.yscale * -1,
                self.ylims[1, 0] * self.yscale
            ])
            widget.vline.set_visible(True)
            widget.hline.set_visible(True)

        # Draw the start and stop lines
        for (i, j), widget in np.ndenumerate(self.axes):

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
            ydata = (-1 * self.ylims[i, j], self.ylims[i, j])
            xdata = (tStart, tStart)
            if np.isnan(xdata).all():
                xdata = (self.configData['responseWindow'][0], self.configData['responseWindow'][0])
            widget.start.set_xdata(xdata)
            widget.start.set_ydata(ydata)
            xdata = (tStop, tStop)
            if np.isnan(xdata).all():
                xdata = (self.configData['responseWindow'][1], self.configData['responseWindow'][1])
            widget.stop.set_xdata(xdata)
            widget.stop.set_ydata(ydata)
            for ln, a, c in zip([widget.start, widget.stop], [aStart, aStop], [cStart, cStop]):
                ln.set_visible(True)
                ln.set_color(c)
                ln.set_alpha(a)

        # Set the x and y limits
        for widget, ylim in zip(self.axes.flatten(), self.ylims.flatten()):
            widget.ax.set_ylim([-1 * ylim * self.yscale, ylim * self.yscale])
            widget.ax.set_xlim(self.xlim)

        # Render subplots
        for widget in self.axes.ravel():
            widget.draw()

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