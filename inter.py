import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QProgressBar, QMenuBar, QMenu, QAction,
    QMessageBox
)


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class ProcessorThread(QThread):
    """
    Hilo que invoca a `mpirun` para procesar las imágenes
    y emite señales para actualizar la barra de progreso.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, input_folder, machinefile, kernel_size):
        super().__init__()
        self.input_folder = input_folder
        self.machinefile = machinefile
        self.kernel_size = kernel_size
        self._stop_requested = False

    def run(self):
        # 1) Verificar que la carpeta de entrada exista y contenga BMP:
        if not os.path.isdir(self.input_folder):
            self.finished.emit("La carpeta de entrada no es válida.")
            return

        all_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        total_images = len(all_files)
        if total_images == 0:
            self.finished.emit("No se encontraron imágenes en la carpeta seleccionada.")
            return

        # 2) Construir el comando para el programa de C:
        #    ./programa <kernel_size> <input_folder>
        command = [
            "mpirun",
            "-n", "11",    # usa tantos procesos MPI como CPUs haya
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
            self.finished.emit(f"Error al iniciar mpirun: {e}")
            return

        # 3) Leer el stdout en tiempo real para actualizar la barra de progreso.
        #    Nos fijamos en las líneas que contienen "Terminó imagen" para contar.
        processed = 0
        for line in process.stdout:
            if self._stop_requested:
                process.kill()
                break

            if "Terminó imagen" in line:
                try:
                    parts = line.strip().split()
                    # ej: "[RANK 2] Terminó imagen 17"
                    idx = int(parts[-1])
                    processed += 1
                    porcentaje = int((processed / total_images) * 100)
                    self.progress.emit(porcentaje)
                except:
                    pass

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self.finished.emit(f"Error en ejecución:\n{stderr}")
            return

        self.progress.emit(100)
        self.finished.emit("Procesamiento completado.")


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento Distribuido de Imágenes")
        self.resize(800, 600)

        # Variables internas
        self.input_folder = ""
        self.machinefile = "machinefile"  # se asume en el cwd
        self.kernel_size = 55
        self.processor_thread = None

        # Carpeta de salida fija en ./salidas/
        self.output_folder = os.path.join(os.getcwd(), "salidas")

        # Construcción del menú
        self._build_menu()

        # Construcción de la UI central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # 1. Selección de Carpeta de Entrada
        input_layout = QHBoxLayout()
        btn_select_input = QPushButton("Seleccionar Carpeta de Imágenes")
        btn_select_input.clicked.connect(self.select_input_folder)
        self.lbl_input_folder = QLabel("No se ha seleccionado carpeta de entrada")
        self.lbl_input_folder.setStyleSheet("color: gray;")
        input_layout.addWidget(btn_select_input)
        input_layout.addWidget(self.lbl_input_folder)
        main_layout.addLayout(input_layout)

        # 2. Mostrar Carpeta de Salida (solo lectura)
        output_layout = QHBoxLayout()
        lbl_output_text = QLabel("Carpeta de Salida:")
        self.lbl_output_folder = QLabel(self.output_folder)
        self.lbl_output_folder.setStyleSheet("color: black; font-weight: bold;")
        output_layout.addWidget(lbl_output_text)
        output_layout.addWidget(self.lbl_output_folder)
        main_layout.addLayout(output_layout)

        # 3. Botón de Inicio de Procesamiento
        self.btn_start = QPushButton("Iniciar Procesamiento")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        main_layout.addWidget(self.btn_start)

        # 4. Barra de Progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # 5. Sección de Métricas (puntos 4-6)
        metrics_layout = QVBoxLayout()
        metrics_header = QLabel("Métricas de Procesamiento")
        font_header = QFont()
        font_header.setPointSize(14)
        font_header.setBold(True)
        metrics_header.setFont(font_header)
        metrics_layout.addWidget(metrics_header)

        self.lbl_total_reads = QLabel("Total localidades leídas: N/A")
        self.lbl_total_writes = QLabel("Total localidades escritas: N/A")
        self.lbl_pixels_per_sec = QLabel("Pixeles por segundo: N/A")
        self.lbl_mips = QLabel("Rendimiento (MIPS): N/A")

        metrics_layout.addWidget(self.lbl_total_reads)
        metrics_layout.addWidget(self.lbl_total_writes)
        metrics_layout.addWidget(self.lbl_pixels_per_sec)
        metrics_layout.addWidget(self.lbl_mips)

        main_layout.addLayout(metrics_layout)

        central_widget.setLayout(main_layout)

        # Timer para refrescar métricas cada 5 segundos
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(5000)
        self.metrics_timer.timeout.connect(self.load_metrics_file)

    def _build_menu(self):
        """
        Crea la barra de menú con la opción “Equipo” y “Ayuda”.
        """
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # Menú “Archivo” (vacío por ahora)
        menu_file = QMenu("Archivo", self)
        menubar.addMenu(menu_file)

        # Menú “Equipo” con integrantes hardcodeados
        menu_team = QMenu("Integrantes", self)
        action_show_team = QAction("Ver Integrantes", self)
        action_show_team.triggered.connect(self.show_team)
        menu_team.addAction(action_show_team)
        menubar.addMenu(menu_team)

        # Menú “Ayuda”
        menu_help = QMenu("Ayuda", self)
        action_about = QAction("Acerca de...", self)
        action_about.triggered.connect(self.show_about)
        menu_help.addAction(action_about)
        menubar.addMenu(menu_help)

    def show_team(self):
        """
        Muestra un cuadro de diálogo con los nombres hardcodeados del equipo.
        """
        integrantes = "Hedguhar Domínguez González - A01730640\nHugo Muñoz Rodríguez - A01736149\nRogelio Hernández Cortés - A01735819"
        QMessageBox.information(self, "Equipo de Desarrollo", integrantes)

    def show_about(self):
        """
        Muestra un cuadro de diálogo “Acerca de” con información de la aplicación.
        """
        QMessageBox.information(
            self, "Acerca de",
            "Procesamiento Distribuido de Imágenes\n"
            "Versión 1.0\n\n"
            "Interfaz desarrollada con PyQt5\n"
            "Generado para el reto de procesamiento de imágenes\n"
        )

    def select_input_folder(self):
        """
        Abre un diálogo para seleccionar la carpeta donde están las imágenes a procesar.
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar Carpeta de Imágenes", "", QFileDialog.ShowDirsOnly
        )
        if folder:
            self.input_folder = folder
            self.lbl_input_folder.setText(folder)
            self.lbl_input_folder.setStyleSheet("color: black;")
            self._update_start_button_state()

    def _update_start_button_state(self):
        """
        Habilita o deshabilita el botón de iniciar procesamiento según
        si ya se ha seleccionado carpeta de entrada.
        """
        if self.input_folder:
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(False)

    def start_processing(self):
        """
        Inicia el hilo de procesamiento distribuido con mpi.
        """
        # Limpiar barra de progreso y métricas antes de iniciar
        self.progress_bar.setValue(0)
        self.lbl_total_reads.setText("Total localidades leídas: N/A")
        self.lbl_total_writes.setText("Total localidades escritas: N/A")
        self.lbl_pixels_per_sec.setText("Pixeles por segundo: N/A")
        self.lbl_mips.setText("Rendimiento (MIPS): N/A")

        # Verificar que existan archivos BMP en la carpeta de entrada
        all_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        if len(all_files) == 0:
            QMessageBox.warning(self, "Sin archivos",
                                "No se encontraron imágenes en la carpeta de entrada.")
            return

        # Crear la carpeta de salida si no existe
        if not os.path.isdir(self.output_folder):
            try:
                os.makedirs(self.output_folder, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"No se pudo crear carpeta de salida:\n{e}")
                return

        # Lanzar el hilo de procesamiento
        self.processor_thread = ProcessorThread(
            input_folder=self.input_folder,
            machinefile=self.machinefile,
            kernel_size=self.kernel_size
        )
        self.processor_thread.progress.connect(self.progress_bar.setValue)
        self.processor_thread.finished.connect(self.on_processing_finished)

        self.metrics_timer.start()  # comenzar a actualizar métricas
        self.processor_thread.start()
        self.btn_start.setEnabled(False)

    def on_processing_finished(self, message):
        """
        Se invoca cuando el hilo de procesamiento finaliza.
        """
        self.metrics_timer.stop()
        self.load_metrics_file()
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Proceso terminado", message)
        self.btn_start.setEnabled(True)

    def load_metrics_file(self):
        """
        Lee `estadisticas.txt` y actualiza las métricas en pantalla.
        """
        if not os.path.exists("estadisticas.txt"):
            return

        try:
            with open("estadisticas.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return

        for line in lines:
            line = line.strip()
            if line.startswith("Total de localidades leídas"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_total_reads.setText(f"Total localidades leídas: {parts[1].strip()}")
            elif line.startswith("Total de localidades escritas"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_total_writes.setText(f"Total localidades escritas: {parts[1].strip()}")
            elif line.startswith("Pixeles procesados por segundo"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_pixels_per_sec.setText(f"Pixeles por segundo: {parts[1].strip()}")
            elif line.startswith("Rendimiento estimado"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_mips.setText(f"Rendimiento (MIPS): {parts[1].strip()}")

        self.metrics_timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

