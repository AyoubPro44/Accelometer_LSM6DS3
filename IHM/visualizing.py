import sys
import math
import time
import pygame
from getting_data import get_current_data

# Cube points centered at origin
CUBE_VERTICES = [
    [-1, -1, -1], [1, -1, -1],
    [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1],
    [1, 1, 1], [-1, 1, 1]
]

EDGES = [
    (0, 1,(255,0,0)), (1, 2,(255,0,0)), (2, 3,(255,0,0)), (3, 0,(255,0,0)),
    (4, 5,(0,255,0)), (5, 6,(0,255,0)), (6, 7,(0,255,0)), (7, 4,(0,255,0)),
    (0, 4,(0,0,255)), (1, 5,(0,0,255)), (2, 6,(0,0,0)), (3, 7,(0,0,0))
]

WIDTH, HEIGHT = 400, 400
CENTER = WIDTH // 2, HEIGHT // 2
SCALE = 100

class CubeVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Orientation Cube")
        self.clock = pygame.time.Clock()
        self.roll = self.pitch = self.yaw = 0
        self.alpha = 0.98
        self.last_time = time.time()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            dt = time.time() - self.last_time
            self.last_time = time.time()
            self.update_orientation(dt)
            self.draw_cube()
            pygame.display.flip()
            self.clock.tick(50)

        pygame.quit()

    def update_orientation(self, dt):
        data = get_current_data()
        if not data:
            return
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = data
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = acc_x * 10, acc_y* 10, acc_z* 10, gyro_x* 10, gyro_y* 10, gyro_z* 10


        # Convert degrees/sec to radians/sec
        gyro_x = math.radians(gyro_x)
        gyro_y = math.radians(gyro_y)
        gyro_z = math.radians(gyro_z)

        # Integrate gyroscope angular velocity to get orientation
        self.roll += gyro_x * dt
        self.pitch += gyro_y * dt
        self.yaw += gyro_z * dt



    def rotate_point(self, x, y, z):
        # Apply rotation around X (roll)
        y, z = y * math.cos(self.roll) - z * math.sin(self.roll), y * math.sin(self.roll) + z * math.cos(self.roll)
        # Apply rotation around Y (pitch)
        x, z = x * math.cos(self.pitch) + z * math.sin(self.pitch), -x * math.sin(self.pitch) + z * math.cos(self.pitch)
        # Apply rotation around Z (yaw)
        x, y = x * math.cos(self.yaw) - y * math.sin(self.yaw), x * math.sin(self.yaw) + y * math.cos(self.yaw)
        return x, y, z

    def draw_cube(self):
        self.screen.fill((255, 255, 255))
        projected = []
        for x, y, z in CUBE_VERTICES:
            x, y, z = self.rotate_point(x, y, z)
            factor = 300 / (300 + z * SCALE)
            x_proj = x * SCALE * factor + CENTER[0]
            y_proj = y * SCALE * factor + CENTER[1]
            projected.append((x_proj, y_proj))

        for start, end,color in EDGES:
            pygame.draw.line(self.screen, color, projected[start], projected[end], 2)