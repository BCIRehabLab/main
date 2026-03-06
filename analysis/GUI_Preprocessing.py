#Preprocessing Pipeline v0.1
#Artur Aharonyan
#BCIRehabLab
#The Catholic University of America

import sys
import mne
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QLabel, QCheckBox, QMessageBox
)

class EEGPreprocessGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MNE EEG Preprocessing GUI")
        self.setGeometry(200, 200, 400, 400)

        self.raw = None

        layout = QVBoxLayout()

        self.file_label = QLabel("No file loaded")
        layout.addWidget(self.file_label)

        self.load_button = QPushButton("Load VHDR File")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        # preprocessing options
        self.bandpass_cb = QCheckBox("Band-pass Filter (1–40 Hz)")
        layout.addWidget(self.bandpass_cb)

        self.notch_cb = QCheckBox("Notch Filter (50/60 Hz)")
        layout.addWidget(self.notch_cb)

        self.ref_cb = QCheckBox("Average Reference")
        layout.addWidget(self.ref_cb)

        self.ica_cb = QCheckBox("Run ICA (artifact removal)")
        layout.addWidget(self.ica_cb)

        self.epoch_cb = QCheckBox("Create Epochs from Events")
        layout.addWidget(self.epoch_cb)

        self.run_button = QPushButton("Run Preprocessing")
        self.run_button.clicked.connect(self.run_pipeline)
        layout.addWidget(self.run_button)

        self.save_button = QPushButton("Save Processed Data")
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def load_file(self):

        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select BrainVision file",
            "",
            "BrainVision (*.vhdr)"
        )

        if file:
            self.raw = mne.io.read_raw_brainvision(file, preload=True)
            self.file_label.setText(f"Loaded:\n{file}")

    def run_pipeline(self):

        if self.raw is None:
            QMessageBox.warning(self, "Error", "Load a file first")
            return

        raw = self.raw

        try:

            if self.bandpass_cb.isChecked():
                raw.filter(1., 40.)

            if self.notch_cb.isChecked():
                raw.notch_filter([50, 60])

            if self.ref_cb.isChecked():
                raw.set_eeg_reference("average")

            if self.ica_cb.isChecked():
                from mne.preprocessing import ICA

                ica = ICA(n_components=20, random_state=97)
                ica.fit(raw)
                raw = ica.apply(raw)

            if self.epoch_cb.isChecked():
                events, event_id = mne.events_from_annotations(raw)
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id,
                    tmin=-0.2,
                    tmax=0.8,
                    preload=True
                )
                self.processed = epochs
            else:
                self.processed = raw

            QMessageBox.information(self, "Success", "Preprocessing complete")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_data(self):

        if not hasattr(self, "processed"):
            QMessageBox.warning(self, "Error", "Run preprocessing first")
            return

        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save file",
            "",
            "FIF (*.fif)"
        )

        if file:
            self.processed.save(file, overwrite=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGPreprocessGUI()
    window.show()
    sys.exit(app.exec())