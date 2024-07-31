/*
 *  moveo.cpp
 *  moveo
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#include "moveo.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <math.h>

Moveo::Moveo() : serial_('$', '#')
{
    connected_ = false;
}

Moveo::~Moveo()
{
    disconnect();
}

bool Moveo::connect(const std::string& port)
{
    if(connected_)
    {
        return true;
    }
    if(!serial_.openPort(port, B115200))
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    SerialPacket pck;
    pck.cmd = CMD_CONNECT;
    pck.dataLen = 0;    
    if(!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket* res = serial_.readPacket(CMD_CONNECT);
    if(res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT  VALID" << std::endl;
        return false;
    }
    delete res;
    connected_ = true;

    return true;
}

void Moveo::disconnect()
{
    std::cout << std::endl
              << std::endl;
    std::cout << "------- STOPPING MOTORS & DISCONNECTING FROM MOVEO -------" << std::endl;
    stop();
    serial_.closePort();
    std::cout << "----------------------------------------------------------" << std::endl;
}

bool Moveo::setMotorParams(std::vector<float> motor_params)
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    SerialPacket pck;
    pck.cmd = CMD_SET_MOTOR_PARAMS;
    pck.dataLen = 48;
    memcpy(&pck.data, motor_params.data(), sizeof(motor_params));

    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_SET_MOTOR_PARAMS);
    if (res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::setMotorRanges(std::vector<int16_t> motor_ranges)
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    SerialPacket pck;
    pck.cmd = CMD_SET_MOTOR_RANGES;
    pck.dataLen = 24;
    for(size_t i=0; i<NUM_MOTORS; ++i)
    {
        pck.data[i*4+0] = (uint8_t)(motor_ranges[2*i] >> 8);
        pck.data[i*4+1] = (uint8_t)(motor_ranges[2*i] & 0xFF);

        pck.data[i*4+2] = (uint8_t)(motor_ranges[2*i+1] >> 8);
        pck.data[i*4+3] = (uint8_t)(motor_ranges[2*i+1] & 0xFF);
    }

    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_SET_MOTOR_RANGES);
    if (res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}


bool Moveo::enable()
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    SerialPacket pck;
    pck.cmd = CMD_ENABLE;
    pck.dataLen = 0;
    
    if(!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket* res = serial_.readPacket(CMD_ENABLE);
    if(res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::disable()
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    
    SerialPacket pck;
    pck.cmd = CMD_DISABLE;
    pck.dataLen = 0;
    
    if(!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket* res = serial_.readPacket(CMD_DISABLE);
    if(res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::stop()
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }

    SerialPacket pck;
    pck.cmd = CMD_STOP;
    pck.dataLen = 0;

    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_STOP);
    if (res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::setTargetPosition(const std::vector<int16_t>& joint_position)
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }
    SerialPacket pck;
    pck.cmd = CMD_SET_TARGET_POSITION;
    pck.dataLen = 12;
    for (size_t i = 0; i < NUM_MOTORS; ++i)
    {
        pck.data[i*2+0] = (uint8_t)(joint_position[i] >> 8);
        pck.data[i*2+1] = (uint8_t)(joint_position[i] & 0xFF);
    }
    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    };

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_SET_TARGET_POSITION);
    if (res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR RECIVED COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

std::vector<int16_t> Moveo::getCurrentPosition()
{
    if(!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return std::vector<int16_t>();
    }

    SerialPacket pck;
    pck.cmd = CMD_GET_CURRENT_POSITION;
    pck.dataLen = 0;
    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return std::vector<int16_t>();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_GET_CURRENT_POSITION);
    if (res == nullptr)
    {
        std::cout << "PACKET WITH POSITION DATA NOT RECIVED OR COMMAND NOT VALID" << std::endl;
        return std::vector<int16_t>();
    }

    if (res->dataLen == 12)
    {
        std::vector<int16_t> joint_position(6);
        joint_position[0] = (int16_t)((uint16_t)res->data[0] << 8 | (uint16_t)res->data[1]);
        joint_position[1] = (int16_t)((uint16_t)res->data[2] << 8 | (uint16_t)res->data[3]);
        joint_position[2] = (int16_t)((uint16_t)res->data[4] << 8 | (uint16_t)res->data[5]);
        joint_position[3] = (int16_t)((uint16_t)res->data[6] << 8 | (uint16_t)res->data[7]);
        joint_position[4] = (int16_t)((uint16_t)res->data[8] << 8 | (uint16_t)res->data[9]);
        joint_position[5] = (int16_t)((uint16_t)res->data[10] << 8 | (uint16_t)res->data[11]);
        delete res;
        return joint_position;
    }
    delete res;

    return std::vector<int16_t>();
}

bool Moveo::closeGripper()
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }

    SerialPacket pck;
    pck.cmd = CMD_CLOSE_GRIPPER;
    pck.dataLen = 0;

    if(!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket* res = serial_.readPacket(CMD_CLOSE_GRIPPER);
    if(res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::openGripper()
{
    if(!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }

    SerialPacket pck;
    pck.cmd = CMD_OPEN_GRIPPER;
    pck.dataLen = 0;

    if(!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket* res = serial_.readPacket(CMD_OPEN_GRIPPER);
    if(res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NOT RECIVED OR COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}

bool Moveo::moveToPosition(const std::vector<float>& motor_params)
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }

    SerialPacket pck;
    pck.cmd = CMD_MOVE_TO_POSITION;
    pck.dataLen = 48;
    memcpy(&pck.data, motor_params.data(), sizeof(float) * NUM_MOTORS * 2);
    
    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    SerialPacket *res = serial_.readPacket(CMD_MOVE_TO_POSITION);
    if (res == nullptr)
    {
        std::cout << "CONFIRMATION PACKET NO RECIVED OR COMMAND NOT VALID" << std::endl;
        return false;
    }
    delete res;

    return true;
}


bool Moveo::trajToPosition(std::vector<float> motor_speeds)
{
    if (!connected_)
    {
        std::cout << "SERIAL PORT NOT CONNECTED" << std::endl;
        return false;
    }

    SerialPacket pck;
    pck.cmd = CMD_TRAJ_TO_POSITION;
    pck.dataLen = 24;
    memcpy(&pck.data, motor_speeds.data(), sizeof(float) * NUM_MOTORS);

    if (!serial_.writePacket(pck))
    {
        std::cout << "UNABLE TO SEND PACKET" << std::endl;
        return false;
    }

    return true;

}