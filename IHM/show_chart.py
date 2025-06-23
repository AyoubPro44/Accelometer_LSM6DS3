import sys
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QHBoxLayout)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from saving_data import *
from getting_data import get_current_data

class SensorChartDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Data Visualization")
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        layout = QVBoxLayout(self)
        
        # Title label
        title = QLabel("Sensor Data - Acceleration & Gyroscope")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create control panel
        control_layout = QHBoxLayout()
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax_accel = self.figure.add_subplot(211)  # Acceleration subplot (top)
        self.ax_gyro = self.figure.add_subplot(212)   # Gyroscope subplot (bottom)
        
        self.setup_chart_style()

        # Initialize data storage for acceleration
        self.x_data, self.y_data, self.z_data = [], [], []
        
        # Initialize data storage for gyroscope
        self.gx_data, self.gy_data, self.gz_data = [], [], []

        # Set up a timer to periodically update the chart
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # Update every 100ms
        
    def setup_chart_style(self):
        """Configure the chart style for acceleration and gyroscope plots"""
        for ax in [self.ax_accel, self.ax_gyro]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_ylim(-7,7)  # Set Y-axis range
            ax.yaxis.set_major_locator(MultipleLocator(10))  # Set ticks every 2 units

        # Label acceleration plot
        self.ax_accel.set_ylabel("Acceleration (°)")
        self.ax_accel.set_title("Acceleration Data")
        
        # Label gyroscope plot
        self.ax_gyro.set_xlabel("Time (s)")
        self.ax_gyro.set_ylabel("Gyroscope (°)")
        self.ax_gyro.set_title("Gyroscope Data")
        
        # Colors and line styles
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    def update_data(self):
        """Fetch new data and update both plots"""
        data = get_current_data()
        
        if data:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = data
            
            # Append new data
            self.x_data.append(accel_x)
            self.y_data.append(accel_y)
            self.z_data.append(accel_z)

            self.gx_data.append(gyro_x)
            self.gy_data.append(gyro_y)
            self.gz_data.append(gyro_z)

            # Plot updated data
            self.plot_data()

    def plot_data(self):
        """Plot both acceleration and gyroscope data"""
        self.setup_chart_style()

        # Plot acceleration data
        self.ax_accel.plot(self.x_data, color=self.colors[0], linewidth=2, label='X Axis')
        self.ax_accel.plot(self.y_data, color=self.colors[1], linewidth=2, label='Y Axis')
        self.ax_accel.plot(self.z_data, color=self.colors[2], linewidth=2, label='Z Axis')
        self.ax_accel.legend(loc='upper right')

        # Plot gyroscope data
        self.ax_gyro.plot(self.gx_data, color=self.colors[0], linewidth=2, label='X Axis')
        self.ax_gyro.plot(self.gy_data, color=self.colors[1], linewidth=2, label='Y Axis')
        self.ax_gyro.plot(self.gz_data, color=self.colors[2], linewidth=2, label='Z Axis')
        self.ax_gyro.legend(loc='upper right')

        # Redraw the canvas
        self.canvas.draw()

def show_current_chart():
    dialog = SensorChartDialog()
    dialog.exec()
