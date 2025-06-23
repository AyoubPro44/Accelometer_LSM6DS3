#include "lm75.h"
#include "stm32f4xx_hal_i2c.h"

#define LM75_ADDRESS		(0x48)

#define LM75_REG_TEMP		(0x00)
#define LM75_REG_CONFIG		(0x01)
#define LM75_REG_THYST		(0x02)
#define LM75_REG_TOS		(0x03)
#define LM75_REG_NONE		(0x04)

// static uint8_t last_reg = LM75_REG_NONE;
extern I2C_HandleTypeDef hi2c1;

int  lm75_read_temp(float *temp)
{
	uint8_t buff[10];
	buff[0]=0x00;
	HAL_I2C_Master_Transmit_IT(&hi2c1, 0x48, buff, 1, 0);
	HAL_I2C_Master_Receive_IT(&hi2c1, 0x48, buff, 2, 0);

	int temp_int = ((uint16_t) buff[0] << 8) | buff[1];
	temp_int >>= 5;

	*temp = (float) temp_int;
	*temp /= 2*2*2;

	return 0;
}



