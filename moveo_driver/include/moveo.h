/*
 *  moveo.h
 *  moveo
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#ifndef __moveo_moveo__
#define __moveo_moveo__

#include "serial_cobs.h"
#include <vector>
#include <functional>

#define NUM_MOTORS 6

class Moveo
{
public: 
    Moveo();
    ~Moveo();

    bool connect(const std::string& port);
    void disconnect();
    bool setMotorParams(std::vector<float> motor_params);
    bool setMotorRanges(std::vector<int16_t> motor_ranges);
    bool enable();
    bool disable();
    bool stop();
    bool setTargetPosition(const std::vector<int16_t>& joint_position);
    std::vector<int16_t> getCurrentPosition();
    bool closeGripper();
    bool openGripper();
    bool moveToPosition(const std::vector<float>& motor_params);
    bool trajToPosition(std::vector<float> motor_speeds);

private:
    SerialCOBS serial_;
    bool connected_;
};

#endif
