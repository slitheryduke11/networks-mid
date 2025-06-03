from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QFileDialog, QListWidget, QProgressBar, QMessageBox
)
import subprocess
import os
import sys

# Supported image formats
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class ProcessorThread(QThread):
    """
    Background thread that launches an external C image-processing program via mpirun,
    monitors the output, and emits progress updates.
    """
    progress = pyqtSignal(int)  # emits integer % of processing progress
    finished = pyqtSignal(str)  # emits final status message

    def __init__(self, input_folder, machinefile="machinefile", kernel_size=55):
        super().__init__()
        self.input_folder = input_folder
        self.machinefile = machinefile
        self.kernel_size = kernel_size

    def run(self):
        """Main thread logic for launching and monitoring the external process."""
        if not os.path.isdir(self.input_folder):
            self.finished.emit("Invalid input folder.")
            return

        all_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        total_images = len(all_files)
        if total_images == 0:
            self.finished.emit("No images found in selected folder.")
            return

        # Command to run the image processing C program with MPI
        command = [
            "mpirun",
            "-n", "11",
            "-f", self.machinefile,
            "./programa",
            str(self.kernel_size),
            self.input_folder
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        except Exception as e:
            self.finished.emit(f"Error launching process: {e}")
            return

        # Read stdout line by line to track processing progress
        processed = 0
        for line in process.stdout:
            if "Terminó imagen" in line:  # Keep this string unless C output is modified
                processed += 1
                percent = int((processed / total_images) * 100)
                self.progress.emit(percent)

        # Final cleanup and error check
        _, stderr = process.communicate()
        if process.returncode != 0:
            self.finished.emit(f"Execution error:\n{stderr}")
            return

        self.progress.emit(100)
        self.finished.emit("Processing completed.")


class DropArea(QLabel):
    """
    Custom QLabel that accepts folder drag-and-drop events.
    """
    folderDropped = pyqtSignal(str)  # emits the folder path dropped

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n Drop image folder here \n\n")
        self.setStyleSheet('''
            QLabel {
                border: 3px dashed #aaa;
                min-height: 250px;
                font-size: 16px;
            }
        ''')
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Accept folder URLs on drag enter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Emit folder path if a directory is dropped."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.folderDropped.emit(path)
                break


class MainWindow(QWidget):
    """
    Main window of the application, combining GUI and process control.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing App")
        self.resize(500, 600)
        self.input_folder = ""
        self.kernel_size = 55
        self.machinefile = "machinefile"

        layout = QVBoxLayout()

        # Title
        self.title_label = QLabel("Distributed Image Processing")
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        # Drag-and-drop area
        self.drop_area = DropArea()
        self.drop_area.folderDropped.connect(self.folder_selected)
        layout.addWidget(self.drop_area)

        # Folder selection display
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: gray")
        layout.addWidget(self.folder_label)

        # Select folder button
        self.select_button = QPushButton("Select Folder")
        self.select_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # Start processing button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)

        # Metric display labels
        self.metrics_labels = {
            "reads": QLabel("Total reads: N/A"),
            "writes": QLabel("Total writes: N/A"),
            "pps": QLabel("Pixels per second: N/A"),
            "mips": QLabel("Performance (MIPS): N/A")
        }
        for label in self.metrics_labels.values():
            layout.addWidget(label)

        self.setLayout(layout)

        # Timer for metric updates
        self.metrics_timer = QTimer()
        self.metrics_timer.setInterval(5000)
        self.metrics_timer.timeout.connect(self.load_metrics_file)

    def folder_selected(self, folder):
        """Triggered when a folder is selected (either dropped or chosen)."""
        self.input_folder = folder
        self.folder_label.setText(folder)
        self.folder_label.setStyleSheet("color: black")
        self.start_button.setEnabled(True)

    def select_folder(self):
        """Open a dialog to manually select an input folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_selected(folder)

    def start_processing(self):
        """Initialize and start the processor thread."""
        self.progress_bar.setValue(0)
        for key, label in self.metrics_labels.items():
            label.setText(label.text().split(":")[0] + ": N/A")

        self.processor = ProcessorThread(self.input_folder, self.machinefile, self.kernel_size)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.processing_finished)
        self.metrics_timer.start()
        self.processor.start()
        self.start_button.setEnabled(False)

    def processing_finished(self, message):
        """Handle the end of the processing task."""
        self.metrics_timer.stop()
        self.load_metrics_file()
        QMessageBox.information(self, "Processing Finished", message)
        self.start_button.setEnabled(True)

    def load_metrics_file(self):
        """Load performance metrics from the file 'estadisticas.txt'."""
        if not os.path.exists("estadisticas.txt"):
            return
        try:
            with open("estadisticas.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return

        for line in lines:
            if line.startswith("Total de localidades leídas"):
                self.metrics_labels["reads"].setText("Total reads: " + line.split(":")[1].strip())
            elif line.startswith("Total de localidades escritas"):
                self.metrics_labels["writes"].setText("Total writes: " + line.split(":")[1].strip())
            elif line.startswith("Pixeles procesados por segundo"):
                self.metrics_labels["pps"].setText("Pixels per second: " + line.split(":")[1].strip())
            elif line.startswith("Rendimiento estimado"):
                self.metrics_labels["mips"].setText("Performance (MIPS): " + line.split(":")[1].strip())


if __name__ == "__main__":
    # Entry point for the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
