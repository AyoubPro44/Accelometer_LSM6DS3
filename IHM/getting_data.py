import time 
import serial  

port = "COM10"
rate = 9600

def get_current_data(PORT=port, rate=rate):
    """Reads the sensor data from the serial port and returns acceleration and gyroscope values."""
    try:
        serial_port = serial.Serial(PORT, rate, timeout=1)
        serial_port.flushInput()
    except serial.SerialException as e:
        print(f"Erreur d'ouverture du port série: {e}")
        return None  

    try:
        for _ in range(10):
            if serial_port.in_waiting > 0:
                line = serial_port.readline().decode("utf-8").strip()
                data_parts = line.split('|')

                if len(data_parts) != 6:
                    print(f"Format de données invalide: {line}")
                    return None  

                try:
                    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = map(float, data_parts)
                    return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
                except ValueError:
                    print(f"Erreur de conversion des données: {line}")
                    return None

            time.sleep(0.1)

    finally:
        serial_port.close()  

    return None
