import os
import subprocess
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QProgressBar, QTableWidget,
    QTableWidgetItem, QMessageBox, QSpinBox,
    QScrollArea
)

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class ProcessorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    log_output = pyqtSignal(str)

    def __init__(self, input_folder, kernel_size, machinefile_path, total_slots):
        super().__init__()
        self.input_folder = input_folder
        self.kernel_size = kernel_size
        self.machinefile_path = machinefile_path
        self.total_slots = total_slots

    def run(self):
        # 1) Validar carpeta de entrada
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

        # 2) Construir comando mpirun
        command = [
            "mpirun",
            "-np", str(self.total_slots),
            "-f", self.machinefile_path,
            "./programa",
            str(self.kernel_size),
            self.input_folder
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        except Exception as e:
            self.finished.emit(f"Error al iniciar mpirun: {e}")
            return

        processed = 0
        # 3) Leer stdout para detectar “Terminó imagen”
        for line in process.stdout:
            self.log_output.emit(line.rstrip("\n"))
            if "Terminó imagen" in line:
                processed += 1
                porcentaje = int((processed / total_images) * 100)
                self.progress.emit(porcentaje)

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self.finished.emit(f"Error en ejecución:\n{stderr}")
            return

        self.progress.emit(100)
        self.finished.emit("Procesamiento completado.")


class DropArea(QLabel):
    """
    Custom QLabel que acepta arrastrar‐y‐soltar carpetas.
    """
    folderDropped = pyqtSignal(str)  # emite la ruta de la carpeta

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n Arrastra aquí la carpeta de imágenes \n\n")
        self.setStyleSheet('''
            QLabel {
                border: 3px dashed #aaa;
                min-height: 200px;
                font-size: 14px;
                color: #555;
            }
        ''')
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Aceptar cuando arrastren URLs de carpetas."""
        if event.mimeData().hasUrls():
            # Verificamos rápidamente si alguna URL es directorio
            for url in event.mimeData().urls():
                if os.path.isdir(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Emitir la carpeta arrastrada (solo la primera)."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.folderDropped.emit(path)
                break


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento Distribuido de Imágenes")
        self.resize(1050, 900)

        # Variables internas
        self.input_folder = ""
        self.kernel_size = 55
        self.machinefile_path = os.path.join(os.getcwd(), "machinefile")
        self.host_table = None
        self.processor_thread = None

        # --- Construcción menú (sin cambios en “Equipo” ni “Ayuda”) ---
        menubar = self.menuBar()
        menu_team = menubar.addMenu("Integrantes")
        action_show_team = menu_team.addAction("Ver Integrantes")
        action_show_team.triggered.connect(self.show_team)
        menu_help = menubar.addMenu("Ayuda")
        action_about = menu_help.addAction("Acerca de...")
        action_about.triggered.connect(self.show_about)

        # --- Construcción UI central ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # 0. Zona de DropArea (arrastrar y soltar carpeta)
        self.drop_area = DropArea()
        self.drop_area.folderDropped.connect(self.on_folder_dropped)
        main_layout.addWidget(self.drop_area)

        # 1. Selector de carpeta de entrada (botón y etiqueta)
        h1 = QHBoxLayout()
        btn_carpeta = QPushButton("Seleccionar Carpeta Imágenes")
        btn_carpeta.clicked.connect(self.select_input_folder)
        self.lbl_input = QLabel("No se ha seleccionado carpeta")
        self.lbl_input.setStyleSheet("color: gray;")
        h1.addWidget(btn_carpeta)
        h1.addWidget(self.lbl_input)
        main_layout.addLayout(h1)

        # 2. Configuración de hosts (QTableWidget)
        lbl_hosts = QLabel("Hosts y slots (uno por línea):")
        lbl_hosts.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(lbl_hosts)

        # Tabla con 2 columnas: “Host” y “Slots”
        self.host_table = QTableWidget(0, 2)
        self.host_table.setHorizontalHeaderLabels(["Host (IP o hostname)", "Slots"])
        self.host_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.host_table)

        # Botones para añadir/quitar filas
        h2 = QHBoxLayout()
        btn_add_row = QPushButton("Añadir host")
        btn_add_row.clicked.connect(self.add_host_row)
        btn_remove_row = QPushButton("Quitar host")
        btn_remove_row.clicked.connect(self.remove_host_row)
        h2.addWidget(btn_add_row)
        h2.addWidget(btn_remove_row)
        main_layout.addLayout(h2)

        # Botón para “Cargar machinefile existente”
        h2b = QHBoxLayout()
        btn_load_mf = QPushButton("Cargar machinefile")
        btn_load_mf.clicked.connect(self.load_existing_machinefile)
        h2b.addWidget(btn_load_mf)
        main_layout.addLayout(h2b)

        # 3. Kernel size
        h3 = QHBoxLayout()
        lbl_kernel = QLabel("Kernel Size:")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(55, 155)
        self.spin_kernel.setValue(self.kernel_size)
        h3.addWidget(lbl_kernel)
        h3.addWidget(self.spin_kernel)
        main_layout.addLayout(h3)

        # 4. Botón iniciar
        self.btn_start = QPushButton("Iniciar Procesamiento")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        main_layout.addWidget(self.btn_start)

        # 5. Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # 6. Métricas
        lbl_metrics = QLabel("Métricas de Procesamiento")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        lbl_metrics.setFont(font)
        main_layout.addWidget(lbl_metrics)

        self.lbl_reads = QLabel("Total localidades leídas: N/A")
        self.lbl_writes = QLabel("Total localidades escritas: N/A")
        self.lbl_pps = QLabel("Pixeles por segundo: N/A")
        self.lbl_mips = QLabel("Rendimiento (MIPS): N/A")
        main_layout.addWidget(self.lbl_reads)
        main_layout.addWidget(self.lbl_writes)
        main_layout.addWidget(self.lbl_pps)
        main_layout.addWidget(self.lbl_mips)

        # Botón para mostrar/ocultar logs
        self.btn_toggle_logs = QPushButton("Ver logs")
        self.btn_toggle_logs.setCheckable(True)
        self.btn_toggle_logs.setChecked(False)
        self.btn_toggle_logs.clicked.connect(self.toggle_logs_visibility)
        main_layout.addWidget(self.btn_toggle_logs)

        # Área de texto para logs (dentro de un scroll)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.txt_logs = QLabel()
        self.txt_logs.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.txt_logs.setStyleSheet("background: black; color: #CCCCCC; font-family: monospace;")
        self.txt_logs.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.txt_logs.setWordWrap(True)
        scroll.setWidget(self.txt_logs)
        scroll.setVisible(False)
        self.scroll_logs = scroll
        main_layout.addWidget(scroll)

        # Timer para recargar métricas cada 5 segundos
        self.metrics_timer = QTimer()
        self.metrics_timer.setInterval(5000)
        self.metrics_timer.timeout.connect(self.load_metrics_file)

        # Cargar machinefile si existe
        self.load_machinefile_if_exists()

        # Mostrar ventana
        self.show()

    def on_folder_dropped(self, folder_path):
        """
        Se dispara cuando el usuario arrastra/solta una carpeta en DropArea.
        Actualiza self.input_folder, la etiqueta y habilita el botón.
        """
        self.input_folder = folder_path
        # Mostrar la carpeta en la etiqueta
        self.lbl_input.setText(folder_path)
        self.lbl_input.setStyleSheet("color: black;")
        self._update_start_button_state()

    def toggle_logs_visibility(self, checked):
        """
        Muestra u oculta el contenedor de logs según el botón.
        """
        if checked:
            self.btn_toggle_logs.setText("Ocultar logs")
            self.scroll_logs.setVisible(True)
        else:
            self.btn_toggle_logs.setText("Ver logs")
            self.scroll_logs.setVisible(False)

    def append_log_line(self, text):
        """
        Va agregando una línea de texto al QLabel que simula la "consola".
        """
        viejo = self.txt_logs.text()
        nuevo = f"{viejo}{text}<br>"
        self.txt_logs.setText(nuevo)
        # Si crece demasiado, recortamos las primeras caracteres
        if len(nuevo) > 20000:
            self.txt_logs.setText(nuevo[-20000:])

    def load_machinefile_if_exists(self):
        """
        Si existe un archivo 'machinefile' en el cwd, lo abre, lo parsea
        y rellena automáticamente la tabla de hosts/slots.
        """
        default_path = os.path.join(os.getcwd(), "machinefile")
        if not os.path.isfile(default_path):
            return  # no hay nada que cargar

        try:
            with open(default_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            QMessageBox.warning(self, "Aviso",
                                f"No se pudo leer el machinefile predeterminado:\n{e}")
            return

        self.host_table.setRowCount(0)
        for raw in lines:
            line = raw.strip()
            if line == "" or ":" not in line:
                continue

            host_part, slots_part = line.split(":", 1)
            host = host_part.strip()
            try:
                slots = int(slots_part.strip())
            except ValueError:
                continue

            row = self.host_table.rowCount()
            self.host_table.insertRow(row)
            self.host_table.setItem(row, 0, QTableWidgetItem(host))
            spin = QSpinBox()
            spin.setRange(1, 128)
            spin.setValue(slots)
            self.host_table.setCellWidget(row, 1, spin)

        self.machinefile_path = default_path
        self._update_start_button_state()

    def load_existing_machinefile(self):
        """
        Permite al usuario elegir un machinefile ya existente,
        lo parsea y rellena la tabla con host:slots.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar machinefile existente",
            os.getcwd(),
            "Todos los archivos (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el archivo:\n{e}")
            return

        self.host_table.setRowCount(0)
        for raw in lines:
            line = raw.strip()
            if line == "":
                continue
            if ":" not in line:
                QMessageBox.warning(self, "Formato incorrecto",
                                    f"La línea «{line}» no tiene formato host:slots.")
                continue
            host_part, slots_part = line.split(":", 1)
            host = host_part.strip()
            try:
                slots = int(slots_part.strip())
            except ValueError:
                QMessageBox.warning(self, "Formato incorrecto",
                                    f"La parte de slots en «{slots_part}» no es un entero válido.")
                continue

            row = self.host_table.rowCount()
            self.host_table.insertRow(row)
            self.host_table.setItem(row, 0, QTableWidgetItem(host))
            spin = QSpinBox()
            spin.setRange(1, 128)
            spin.setValue(slots)
            self.host_table.setCellWidget(row, 1, spin)

        self.machinefile_path = path
        self._update_start_button_state()

    def add_host_row(self):
        """Añade una fila vacía a la tabla de hosts."""
        row = self.host_table.rowCount()
        self.host_table.insertRow(row)
        self.host_table.setItem(row, 0, QTableWidgetItem(""))
        spin = QSpinBox()
        spin.setRange(1, 128)
        spin.setValue(1)
        self.host_table.setCellWidget(row, 1, spin)
        self._update_start_button_state()

    def remove_host_row(self):
        """Quita la fila seleccionada (o la última si nada está seleccionado)."""
        selected = self.host_table.currentRow()
        if selected < 0:
            selected = self.host_table.rowCount() - 1
        if selected >= 0:
            self.host_table.removeRow(selected)
        self._update_start_button_state()

    def select_input_folder(self):
        """Abre un diálogo para seleccionar la carpeta de imágenes."""
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar Carpeta de Imágenes", "", QFileDialog.ShowDirsOnly
        )
        if folder:
            self.input_folder = folder
            self.lbl_input.setText(folder)
            self.lbl_input.setStyleSheet("color: black;")
            self._update_start_button_state()

    def _update_start_button_state(self):
        """
        Habilita “Iniciar Procesamiento” sólo si hay carpeta + al menos un host válido.
        """
        if not self.input_folder:
            self.btn_start.setEnabled(False)
            return
        if self.host_table.rowCount() == 0:
            self.btn_start.setEnabled(False)
            return
        # Verificar que las celdas de “host” no estén vacías
        for r in range(self.host_table.rowCount()):
            item = self.host_table.item(r, 0)
            if not item or item.text().strip() == "":
                self.btn_start.setEnabled(False)
                return
        self.btn_start.setEnabled(True)

    def start_processing(self):
        # 1) Reiniciar barras y métricas
        self.progress_bar.setValue(0)
        self.lbl_reads.setText("Total localidades leídas: N/A")
        self.lbl_writes.setText("Total localidades escritas: N/A")
        self.lbl_pps.setText("Pixeles por segundo: N/A")
        self.lbl_mips.setText("Rendimiento (MIPS): N/A")

        # 2) Validar carpeta de entrada
        if not os.path.isdir(self.input_folder):
            QMessageBox.critical(self, "Error", "Carpeta de entrada no válida.")
            return

        # 3) Verificar que haya imágenes dentro
        img_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]
        if len(img_files) == 0:
            QMessageBox.warning(self, "Sin archivos",
                                "No se encontraron imágenes en la carpeta de entrada.")
            return

        # 4) Leer tabla de hosts y generar “machinefile”
        filas = self.host_table.rowCount()
        total_slots = 0
        try:
            with open(self.machinefile_path, "w", encoding="utf-8") as mf:
                for r in range(filas):
                    host_item = self.host_table.item(r, 0)
                    host_ip = host_item.text().strip()
                    spin = self.host_table.cellWidget(r, 1)
                    slots = spin.value()
                    mf.write(f"{host_ip}:{slots}\n")
                    total_slots += slots
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo crear machinefile:\n{e}")
            return

        # 5) Construir y lanzar el hilo de procesamiento
        self.processor_thread = ProcessorThread(
            input_folder=self.input_folder,
            kernel_size=self.spin_kernel.value(),
            machinefile_path=self.machinefile_path,
            total_slots=total_slots
        )
        self.processor_thread.progress.connect(self.progress_bar.setValue)
        self.processor_thread.finished.connect(self.on_processing_finished)
        self.processor_thread.log_output.connect(self.append_log_line)
        self.txt_logs.setText("")

        self.metrics_timer.start()
        self.processor_thread.start()
        self.btn_start.setEnabled(False)

    def on_processing_finished(self, message):
        self.metrics_timer.stop()
        self.load_metrics_file()
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Proceso terminado", message)
        self.btn_start.setEnabled(True)

    def load_metrics_file(self):
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
                    self.lbl_reads.setText(f"Total localidades leídas: {parts[1].strip()}")
            elif line.startswith("Total de localidades escritas"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_writes.setText(f"Total localidades escritas: {parts[1].strip()}")
            elif line.startswith("Pixeles procesados por segundo"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_pps.setText(f"Pixeles por segundo: {parts[1].strip()}")
            elif line.startswith("Rendimiento estimado"):
                parts = line.split(":")
                if len(parts) == 2:
                    self.lbl_mips.setText(f"Rendimiento (MIPS): {parts[1].strip()}")

        self.metrics_timer.stop()

    def show_team(self):
        """
        Muestra un cuadro de diálogo con los nombres hardcodeados del equipo.
        """
        integrantes = (
            "Hedguhar Domínguez González - A01730640\n"
            "Hugo Muñoz Rodríguez - A01736149\n"
            "Rogelio Hernández Cortés - A01735819"
        )
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
            "Permite configurar hosts y slots dinámicamente."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())

