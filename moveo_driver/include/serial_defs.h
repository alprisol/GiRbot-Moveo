/*
 *  serial_defs.h
 *  moveo
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#ifndef __moveo_serial_defs__
#define __moveo_serial_defs__

#include <cstdint>

typedef enum __attribute__((__packed__))
{
    CMD_CONNECT = 0x00,       // NO Data
    CMD_SET_MOTOR_PARAMS,     // IN Data(48): 6 x 4 Velocity of each motor (float) + 6 x 4 Acceleration of the motors. (float)
    CMD_SET_MOTOR_RANGES,     // IN Data(24): 6 x 2 x 2 Min and Max position of each motor (int16_t)
    CMD_ENABLE,               // NO Data
    CMD_DISABLE,              // NO Data
    CMD_STOP,                 // NO Data
    CMD_SET_TARGET_POSITION,  // IN Data(12): 6 x 2 bytes per position (int16_t)
    CMD_GET_CURRENT_POSITION, // OUT Data(12): 6 x 2 bytes per position (int16_t)
    CMD_CLOSE_GRIPPER,        // NO Data
    CMD_OPEN_GRIPPER,         // NO Data
    CMD_MOVE_TO_POSITION,     // IN Data(48): 6 x 4 Velocity of each motor (float) + 6 x 4 Acceleration of the motors. (float)
    CMD_TRAJ_TO_POSITION,     // IN Data(24): 6 x 4 bytes per velocity (float)
    CMD_RESET = 0xFD,         // IN Data(1): 1B->0xCD
    CMD_INVALID = 0xFF        // NO Data
} CommandCode;

static inline CommandCode byte2Command(const uint8_t b)
{
    if (b >= (uint8_t)CMD_CONNECT && b <= (uint8_t)CMD_TRAJ_TO_POSITION)
        return (CommandCode)b;
    else if (b == (uint8_t)CMD_RESET)
        return CMD_RESET;
    else
        return CMD_INVALID;
}

#endif