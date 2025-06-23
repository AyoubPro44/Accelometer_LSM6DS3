from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
)
from PyQt6.QtGui import QColor, QFont, QBrush
from PyQt6.QtCore import Qt
from database import Database

class SensorDataTable(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Data Table")
        self.setGeometry(300, 200, 900, 600)

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("SENSOR DATA TABLE")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #333; margin-bottom: 20px;")
        layout.addWidget(title)

        # Create table widget
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)
        
        # Load and style table
        self.load_data()
        self.style_table()

    def load_data(self):
        """Load the sensor data from database"""
        db = Database()
        rows = db.get()
        headers = ["ID", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]

        # Set up the table dimensions
        self.table_widget.setRowCount(len(rows))
        self.table_widget.setColumnCount(len(headers))
        
        # Set headers
        self.table_widget.setHorizontalHeaderLabels(headers)
        
        # Populate with database values (no formatting)
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_widget.setItem(row_idx, col_idx, item)

    def style_table(self):
        """Apply styling to the table"""
        self.table_widget.setFont(QFont("Arial", 10))
        
        # General table styling
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: white;
                color: #333;
                gridline-color: #E0E0E0;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12px;
                border: none;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #81C784;
                color: white;
            }
        """)
        
        # Header styling
        header = self.table_widget.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Hide vertical header
        self.table_widget.verticalHeader().setVisible(False)
        
        # Set row heights
        for row in range(self.table_widget.rowCount()):
            self.table_widget.setRowHeight(row, 40)
        
        # Color the first column differently
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item:
                item.setBackground(QColor(245, 245, 245))
                item.setForeground(QBrush(QColor(51, 51, 51)))
                item.setFont(QFont("Arial", 10, QFont.Weight.Bold))


def show_db():
    dialog = SensorDataTable()
    dialog.exec()