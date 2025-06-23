import sys
import time
import serial
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from database import Database
from getting_data import get_current_data

# Flag to track saving status
is_saving = False

class SavingThread(QThread):
    """Thread to save data in the background."""
    data_saved = pyqtSignal(float, float, float, float, float, float)

    def run(self):
        """Runs the data-saving process."""
        global is_saving
        database = Database()

        try:
            while is_saving:
                data = get_current_data()
                if data:
                    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = data
                    # Insert the data into the database
                    database.insert(a_x=accel_x, a_y=accel_y, a_z=accel_z, g_x=gyro_x, g_y=gyro_y, g_z=gyro_z)
                    self.data_saved.emit(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in saving data: {e}")
            is_saving = False
            print("Data saving stopped.")

class RedDotIndicator(QLabel):
    """A custom widget to display a red dot as an indicator."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.setStyleSheet("background-color: red; border-radius: 15px;")
        
    def paintEvent(self, event):
        """Overrides the default paint event to draw the red dot."""
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(255, 0, 0)))  # Red color
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())

class SavingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Saving Data")
        self.setGeometry(300, 300, 300, 150)

        # Set dark background and transparent effect if needed
        self.setStyleSheet("""
            background-color: rgba(18, 18, 18, 180);
            border-radius: 10px;
        """)

        # Layout for the dialog
        layout = QVBoxLayout()

        # Create a horizontal layout for the red dot and the stop button
        indicator_layout = QHBoxLayout()

        # Red Dot Indicator (Circular Red)
        self.red_dot_label = RedDotIndicator(self)
        indicator_layout.addWidget(self.red_dot_label)

        # Stop Saving Button with new styling
        self.stop_button = QPushButton("Stop Saving", self)
        self.stop_button.setFont(QFont("Arial", 12))
        self.stop_button.setFixedHeight(40)
        self.stop_button.clicked.connect(self.stop_saving)

        # Button styling
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4C4C;  /* Light red */
                color: white;
                border-radius: 15px;
                border: 2px solid #FF0000;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #FF1A1A;  /* Darker red */
            }
            QPushButton:pressed {
                background-color: #D50000;  /* Dark red when pressed */
            }
        """)

        indicator_layout.addWidget(self.stop_button)

        # Add the indicator_layout to the main layout
        layout.addLayout(indicator_layout)

        self.setLayout(layout)

    def stop_saving(self):
        """Stops the saving process and closes the dialog."""
        global is_saving
        is_saving = False
        self.red_dot_label.setStyleSheet("background-color: red; border-radius: 15px;")
        self.close()

def save_data():
    """Saves the data to the database and shows the saving dialog."""
    global is_saving

    # Create the saving dialog
    dialog = SavingDialog()

    # Start the saving thread
    saving_thread = SavingThread()
    saving_thread.data_saved.connect(lambda accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z: print(f"Data saved: {accel_x}, {accel_y}, {accel_z}, {gyro_x}, {gyro_y}, {gyro_z}"))
    
    # Start saving process in a background thread
    is_saving = True
    saving_thread.start()

    # Show the dialog and run until it is closed
    dialog.exec()

    # Stop the saving process when the dialog is closed
    is_saving = False
    saving_thread.wait()
    dialog.close()
