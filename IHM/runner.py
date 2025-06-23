from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QGraphicsDropShadowEffect
from PyQt6.QtGui import QPixmap, QFont, QColor, QMovie
from PyQt6.QtCore import Qt, QTimer

from getting_data import get_current_data
from prediction import predict_motion


class MotionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 350)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # === Image / Animation ===
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(220)
        main_layout.addWidget(self.image_label)

        # === Sensor Group Box ===
        sensor_group = QGroupBox("📡 Sensor Readings")
        sensor_group.setStyleSheet("""
            QGroupBox {
                color: #00c6ff;
                font-weight: bold;
                font-size: 11pt;
                border: 1px solid #00c6ff;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
                background-color: #1a1a1d;
            }
        """)
        sensor_layout = QGridLayout()
        sensor_layout.setSpacing(10)
        sensor_group.setLayout(sensor_layout)
        main_layout.addWidget(sensor_group)

        # === Accelerometer Labels ===
        sensor_layout.addWidget(QLabel("Accelerometer (g):"), 0, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        self.accel_x_label = QLabel("X: 0.00")
        self.accel_y_label = QLabel("Y: 0.00")
        self.accel_z_label = QLabel("Z: 0.00")

        for lbl in (self.accel_x_label, self.accel_y_label, self.accel_z_label):
            lbl.setFont(QFont("Consolas", 10))
            lbl.setStyleSheet("color: #00c6ff;")

        sensor_layout.addWidget(self.accel_x_label, 1, 0)
        sensor_layout.addWidget(self.accel_y_label, 1, 1)
        sensor_layout.addWidget(self.accel_z_label, 1, 2)

        # === Gyroscope Labels ===
        sensor_layout.addWidget(QLabel("Gyroscope (°/s):"), 2, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        self.gyro_x_label = QLabel("X: 0.00")
        self.gyro_y_label = QLabel("Y: 0.00")
        self.gyro_z_label = QLabel("Z: 0.00")

        for lbl in (self.gyro_x_label, self.gyro_y_label, self.gyro_z_label):
            lbl.setFont(QFont("Consolas", 10))
            lbl.setStyleSheet("color: #00c6ff;")

        sensor_layout.addWidget(self.gyro_x_label, 3, 0)
        sensor_layout.addWidget(self.gyro_y_label, 3, 1)
        sensor_layout.addWidget(self.gyro_z_label, 3, 2)

        # === State label ===
        self.state_label = QLabel("🧠 State: Unknown", self)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.state_label.setStyleSheet("""
            background-color: #25252a;
            color: #ff9d00;
            border-radius: 10px;
            padding: 12px;
        """)
        main_layout.addWidget(self.state_label)

        # === Glowing Shadow Effect ===
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(25)
        glow.setOffset(0)
        glow.setColor(QColor(0, 170, 255, 160))
        self.setGraphicsEffect(glow)

        self.setLayout(main_layout)

        # Image and GIF files
        self.image_files = {
            "walking": "pika_walk.gif",
            "running": "pika_run.gif",
            "standing": "stand.png"
        }

        # Movie objects for GIFs
        self.movies = {
            "walking": QMovie(self.image_files["walking"]),
            "running": QMovie(self.image_files["running"])
        }

        # Current state
        self.current_state = None
        self.set_image("standing")

        # Timer to update data
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(500)

    def set_image(self, state):
        if self.current_state == state:
            return

        self.current_state = state
        image_path = self.image_files.get(state)

        if state in self.movies:
            movie = self.movies[state]
            movie.setScaledSize(self.image_label.size())
            self.image_label.setMovie(movie)
            movie.start()
        else:
            for movie in self.movies.values():
                movie.stop()
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        for movie in self.movies.values():
            movie.setScaledSize(self.image_label.size())
        return super().resizeEvent(event)

    def update_loop(self):
        data = get_current_data()
        if data:
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = data

            # Update accel labels
            self.accel_x_label.setText(f"X: {accel_x:.2f}")
            self.accel_y_label.setText(f"Y: {accel_y:.2f}")
            self.accel_z_label.setText(f"Z: {accel_z:.2f}")

            # Update gyro labels
            self.gyro_x_label.setText(f"X: {gyro_x:.2f}")
            self.gyro_y_label.setText(f"Y: {gyro_y:.2f}")
            self.gyro_z_label.setText(f"Z: {gyro_z:.2f}")

            state = predict_motion(accel_x, accel_y, accel_z).lower()
            self.state_label.setText(f"🧠 State: {state.capitalize()}")
            self.set_image(state)
