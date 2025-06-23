import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QGraphicsDropShadowEffect, QLabel, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

# Dummy imports – replace with your actual modules
from saving_data import save_data
from show_database_table import show_db
from show_chart import show_current_chart
from visualizing import CubeVisualizer
from runner import MotionWidget


class ModernInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("⚡ Ultra-Modern Dashboard")
        self.setGeometry(200, 200, 1000, 600)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setStyleSheet("""
            QWidget {
                background-color: #0e0e10;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00c6ff,
                    stop:1 #0072ff
                );
                color: white;
                padding: 12px;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5ec6ff,
                    stop:1 #339dff
                );
            }
            QPushButton:pressed {
                background-color: #004ba0;
            }
            QLabel {
                color: #cfcfcf;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #18181c;
                border: 1px solid #444;
                border-radius: 15px;
                padding: 15px;
                margin-top: 15px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #ff9d00;
                font-size: 15px;
                font-weight: bold;
            }
        """)

        main_layout = QHBoxLayout(self)

        # ==== LEFT PANEL ====
        left_panel = QVBoxLayout()
        button_config = {
            "🚀 Start Saving Data": self.start_saving_data,
            "🧊 3D Visualizing": self.visualizing,
            "📁 Show DB Table": self.show_db_table,
            "📈 Show Chart": self.show_chart,
        }

        for label, handler in button_config.items():
            button = QPushButton(label)
            button.setFixedHeight(45)
            button.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))

            # Glowing effect
            glow = QGraphicsDropShadowEffect()
            glow.setBlurRadius(20)
            glow.setOffset(0)
            glow.setColor(QColor(0, 188, 255, 160))
            button.setGraphicsEffect(glow)

            button.clicked.connect(handler)
            left_panel.addWidget(button)
            left_panel.addSpacing(12)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # ==== RIGHT PANEL ====
        self.motion_widget = MotionWidget()
        self.motion_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.motion_widget, 3)

    # === Button Handlers ===
    def start_saving_data(self):
        # Removed port and baudrate references
        save_data()

    def visualizing(self):
        CubeVisualizer().run()

    def show_db_table(self):
        show_db()

    def show_chart(self):
        show_current_chart()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark Palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    app.setPalette(dark_palette)

    # Launch window
    window = ModernInterface()
    window.show()
    sys.exit(app.exec())
