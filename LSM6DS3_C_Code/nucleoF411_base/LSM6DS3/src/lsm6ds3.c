/*
 * lsm6ds3.c
 *
 *  Created on: Feb 26, 2025
 *      Author: Pavilion
 */

#include "lsm6ds3.h"

void LSM6DS3_Init(LSM6DS3_HandleTypeDef *dev, I2C_HandleTypeDef *hi2c) {
    dev->hi2c = hi2c;
}

uint8_t LSM6DS3_WhoAmI(LSM6DS3_HandleTypeDef *dev) {
    uint8_t who_am_i = 0;
    LSM6DS3_ReadReg(dev, 0x0F, &who_am_i);
    return who_am_i;
}

HAL_StatusTypeDef LSM6DS3_WriteReg(LSM6DS3_HandleTypeDef *dev, uint8_t reg, uint8_t value) {
    uint8_t data[2] = {reg, value};
    return HAL_I2C_Master_Transmit_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, data, 2, HAL_MAX_DELAY);
}

HAL_StatusTypeDef LSM6DS3_ReadReg(LSM6DS3_HandleTypeDef *dev, uint8_t reg, uint8_t *value) {
    HAL_StatusTypeDef ret;
    ret = HAL_I2C_Master_Transmit_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS , &reg, 1, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;
    return HAL_I2C_Master_Receive_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, value, 1, HAL_MAX_DELAY);
}

HAL_StatusTypeDef LSM6DS3_SetAccelerometerConfig(LSM6DS3_HandleTypeDef *dev, LSM6DS3_OutputDataRate odr, LSM6DS3_AccelerometerFullScale fs) {
    return LSM6DS3_WriteReg(dev, LSM6DS3_REG_CTRL1_XL, odr | fs);
}

HAL_StatusTypeDef LSM6DS3_SetGyroscopeConfig(LSM6DS3_HandleTypeDef *dev, LSM6DS3_OutputDataRate odr, LSM6DS3_GyroscopeFullScale fs) {
    return LSM6DS3_WriteReg(dev, LSM6DS3_REG_CTRL2_G, odr | fs);
}

HAL_StatusTypeDef LSM6DS3_ReadAccelerometer(LSM6DS3_HandleTypeDef *dev, float *x, float *y, float *z) {
    uint8_t accel_data[6];
    HAL_StatusTypeDef ret;
    int16_t raw_x, raw_y, raw_z;
    float sensitivity = 0.061f;

    ret = HAL_I2C_Master_Transmit_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, (uint8_t[]){LSM6DS3_REG_OUT_X_L_XL}, 1, 0);
    if (ret != HAL_OK) return ret;
    ret = HAL_I2C_Master_Receive_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, accel_data, 6, 0);
    if (ret != HAL_OK) return ret;

    raw_x = (int16_t)((accel_data[1] << 8) | accel_data[0]);
    raw_y = (int16_t)((accel_data[3] << 8) | accel_data[2]);
    raw_z = (int16_t)((accel_data[5] << 8) | accel_data[4]);

    // Convert to g (1g = 9.81 m/s²)
    *x = (raw_x * sensitivity) / 1000.0f * 9.81f;
    *y = (raw_y * sensitivity) / 1000.0f * 9.81f;
    *z = (raw_z * sensitivity) / 1000.0f * 9.81f;

    *x = *x /10;
    *y = *y / 10;
    *z = *z /10;

    return HAL_OK;
}

HAL_StatusTypeDef LSM6DS3_ReadGyroscope(LSM6DS3_HandleTypeDef *dev, float *x, float *y, float *z) {
    uint8_t gyros_data[6];
    HAL_StatusTypeDef ret;
    int16_t raw_x, raw_y, raw_z;
    float sensitivity = 4.375f;

    ret = HAL_I2C_Master_Transmit_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, (uint8_t[]){LSM6DS3_REG_OUT_X_L_G}, 1, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;
    ret = HAL_I2C_Master_Receive_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, gyros_data, 6, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    raw_x = (int16_t)((gyros_data[1] << 8) | gyros_data[0]);
    raw_y = (int16_t)((gyros_data[3] << 8) | gyros_data[2]);
    raw_z = (int16_t)((gyros_data[5] << 8) | gyros_data[4]);

    *x = raw_x * sensitivity / 1000;
    *y = raw_y * sensitivity / 1000;
    *z = raw_z * sensitivity / 1000;

    *x = *x /10;
    *y = *y / 10;
    *z = *z /10;

    return HAL_OK;
}



HAL_StatusTypeDef LSM6DS3_ReadTemperature(LSM6DS3_HandleTypeDef *dev, float *temp) {
    HAL_StatusTypeDef ret;
    uint8_t temp_vals[2];
    int16_t temp_int;

    ret = HAL_I2C_Master_Transmit_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, (uint8_t[]){LSM6DS3_REG_OUT_TEMP_L}, 1, 0);
    if (ret != HAL_OK) return ret;
    ret = HAL_I2C_Master_Receive_IT(dev->hi2c, LSM6DS3_I2C_ADDRESS, temp_vals, 2, 0);
    if (ret != HAL_OK) return ret;

    temp_int = (int16_t)((temp_vals[1] << 8) | temp_vals[0]);
    temp_int >>= 4;
    *temp = (float) temp_int / 16.0f + 25.0f;

    return HAL_OK;
}
