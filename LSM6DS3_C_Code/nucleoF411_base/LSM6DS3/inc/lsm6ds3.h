// lsm6ds3.h
#ifndef LSM6DS3_H_
#define LSM6DS3_H_

#include "stm32f4xx_hal.h"
#include "stm32f4xx_hal_i2c.h"


#define LSM6DS3_I2C_ADDRESS 0x6A



typedef struct {
    I2C_HandleTypeDef *hi2c;
} LSM6DS3_HandleTypeDef;


typedef enum {
    LSM6DS3_REG_FUNC_CFG_ACCESS = 0x01,
    LSM6DS3_REG_SLV0_ADD = 0x02,
    LSM6DS3_REG_SLV0_SUBADDR = 0x03,
    LSM6DS3_REG_SLV0_CONFIG = 0x04,
    LSM6DS3_REG_SLV1_ADD = 0x05,
    LSM6DS3_REG_SLV1_SUBADDR = 0x06,
    LSM6DS3_REG_SLV1_CONFIG = 0x07,
    LSM6DS3_REG_SLV2_ADD = 0x08,
    LSM6DS3_REG_SLV2_SUBADDR = 0x09,
    LSM6DS3_REG_SLV2_CONFIG = 0x0A,
    LSM6DS3_REG_SLV3_ADD = 0x0B,
    LSM6DS3_REG_SLV3_SUBADDR = 0x0C,
    LSM6DS3_REG_SLV3_CONFIG = 0x0D,
    LSM6DS3_REG_DATAWRITE_ADD = 0x0E,
    LSM6DS3_REG_WHO_AM_I = 0x0F,
    LSM6DS3_REG_CTRL1_XL = 0x10,
    LSM6DS3_REG_CTRL2_G = 0x11,
    LSM6DS3_REG_CTRL3_C = 0x12,
    LSM6DS3_REG_CTRL4_C = 0x13,
    LSM6DS3_REG_CTRL5_C = 0x14,
    LSM6DS3_REG_CTRL6_C = 0x15,
    LSM6DS3_REG_CTRL7_G = 0x16,
    LSM6DS3_REG_CTRL8_XL = 0x17,
    LSM6DS3_REG_CTRL9_XL = 0x18,
    LSM6DS3_REG_CTRL10_C = 0x19,
    LSM6DS3_REG_MASTER_CONFIG = 0x1A,
    LSM6DS3_REG_WAKE_UP_SRC = 0x1B,
    LSM6DS3_REG_TAP_SRC = 0x1C,
    LSM6DS3_REG_D6D_SRC = 0x1D,
    LSM6DS3_REG_STATUS_REG = 0x1E,
    LSM6DS3_REG_OUT_TEMP_L = 0x20,
    LSM6DS3_REG_OUT_TEMP_H = 0x21,
    LSM6DS3_REG_OUT_X_L_G = 0x22,
    LSM6DS3_REG_OUT_X_H_G = 0x23,
    LSM6DS3_REG_OUT_Y_L_G = 0x24,
    LSM6DS3_REG_OUT_Y_H_G = 0x25,
    LSM6DS3_REG_OUT_Z_L_G = 0x26,
    LSM6DS3_REG_OUT_Z_H_G = 0x27,
    LSM6DS3_REG_OUT_X_L_XL = 0x28,
    LSM6DS3_REG_OUT_X_H_XL = 0x29,
    LSM6DS3_REG_OUT_Y_L_XL = 0x2A,
    LSM6DS3_REG_OUT_Y_H_XL = 0x2B,
    LSM6DS3_REG_OUT_Z_L_XL = 0x2C,
    LSM6DS3_REG_OUT_Z_H_XL = 0x2D,
    LSM6DS3_REG_FIFO_CTRL1 = 0x06,
    LSM6DS3_REG_FIFO_CTRL2 = 0x07,
    LSM6DS3_REG_FIFO_CTRL3 = 0x08,
    LSM6DS3_REG_FIFO_CTRL4 = 0x09,
    LSM6DS3_REG_FIFO_CTRL5 = 0x0A,
    LSM6DS3_REG_INT1_CTRL = 0x0D,
    LSM6DS3_REG_INT2_CTRL = 0x0E,
    LSM6DS3_REG_MD1_CFG = 0x5E,
    LSM6DS3_REG_MD2_CFG = 0x5F,
    LSM6DS3_REG_TAP_CFG = 0x58,
    LSM6DS3_REG_WAKE_UP_THS = 0x5B,
    LSM6DS3_REG_FREE_FALL = 0x5D
} LSM6DS3_RegisterAddress;

typedef enum {
    LSM6DS3_ODR_OFF = 0x00,
    LSM6DS3_ODR_12_5HZ = 0x10,
    LSM6DS3_ODR_26HZ = 0x20,
    LSM6DS3_ODR_52HZ = 0x30,
    LSM6DS3_ODR_104HZ = 0x40,
    LSM6DS3_ODR_208HZ = 0x50,
    LSM6DS3_ODR_416HZ = 0x60,
    LSM6DS3_ODR_833HZ = 0x70,
    LSM6DS3_ODR_1660HZ = 0x80,
    LSM6DS3_ODR_3330HZ = 0x90,
    LSM6DS3_ODR_6660HZ = 0xA0
} LSM6DS3_OutputDataRate;

typedef enum {
    LSM6DS3_FS_2G = 0x00,
    LSM6DS3_FS_4G = 0x08,
    LSM6DS3_FS_8G = 0x0C,
    LSM6DS3_FS_16G = 0x04
} LSM6DS3_AccelerometerFullScale;

typedef enum {
    LSM6DS3_FS_125DPS = 0x02,
    LSM6DS3_FS_250DPS = 0x00,
    LSM6DS3_FS_500DPS = 0x04,
    LSM6DS3_FS_1000DPS = 0x08,
    LSM6DS3_FS_2000DPS = 0x0C
} LSM6DS3_GyroscopeFullScale;


void LSM6DS3_Init(LSM6DS3_HandleTypeDef *dev, I2C_HandleTypeDef *hi2c);
uint8_t LSM6DS3_WhoAmI(LSM6DS3_HandleTypeDef *dev);
HAL_StatusTypeDef LSM6DS3_WriteReg(LSM6DS3_HandleTypeDef *dev, uint8_t reg, uint8_t value);
HAL_StatusTypeDef LSM6DS3_ReadReg(LSM6DS3_HandleTypeDef *dev, uint8_t reg, uint8_t *value);
HAL_StatusTypeDef LSM6DS3_SetAccelerometerConfig(LSM6DS3_HandleTypeDef *dev, LSM6DS3_OutputDataRate odr, LSM6DS3_AccelerometerFullScale fs);
HAL_StatusTypeDef LSM6DS3_SetGyroscopeConfig(LSM6DS3_HandleTypeDef *dev, LSM6DS3_OutputDataRate odr, LSM6DS3_GyroscopeFullScale fs);
HAL_StatusTypeDef LSM6DS3_ReadAccelerometer(LSM6DS3_HandleTypeDef *dev, float *x, float *y, float *z);
HAL_StatusTypeDef LSM6DS3_ReadTemperature(LSM6DS3_HandleTypeDef *dev, float *temp);
HAL_StatusTypeDef LSM6DS3_ReadGyroscope(LSM6DS3_HandleTypeDef *dev, float *x, float *y, float *z);

#endif /* LSM6DS3_LSM6DS3_H_ */
