# 1 "/home/cirs/moveo/moveo_firmware/moveo_firmware.ino"
# 2 "/home/cirs/moveo/moveo_firmware/moveo_firmware.ino" 2
# 3 "/home/cirs/moveo/moveo_firmware/moveo_firmware.ino" 2
# 4 "/home/cirs/moveo/moveo_firmware/moveo_firmware.ino" 2




typedef enum MotionType
{
  STILL,
  JOINT_INTERP,
  SPACE_INTERP
};

bool connected;
SerialPacket tx_pck;
SerialPacket rx_pck;
uint8_t pck_rx_buffer[128 /*??*/];
uint8_t pck_rx_index;
float motor_maxVel[6 /* Define the number of motors*/];
float motor_maxAccel[6 /* Define the number of motors*/];
int16_t motor_ranges[6 /* Define the number of motors*/ * 2];
int16_t motor_pos[6 /* Define the number of motors*/];
float move_maxVel[6 /* Define the number of motors*/];
float move_Accel[6 /* Define the number of motors*/];
float traj_Vel[6 /* Define the number of motors*/];
MotionType motion_type = STILL;

bool still_first_time = true;

void setup()
{
  // Initialize system upon power-up.
  Serial.begin(115200);

  // Initialize Motor Params lists.
  for (int i = 0; i < 6 /* Define the number of motors*/; ++i)
  {
    motor_maxVel[i] = 3000;
    motor_maxAccel[i] = 3000;
  }

  // Communication
  while (Serial.available() > 0)
  {
    Serial.read();
  }; // Clear serial read buffer
  pck_rx_index = 0;
  connected = false;
}

void loop()
{
  // Process communication
  uint8_t rx_count = Serial.available();
  for (uint8_t i = 0; i < rx_count; ++i)
  {
    pck_rx_buffer[pck_rx_index] = Serial.read();

    if (pck_rx_buffer[pck_rx_index] == 0x00)
    {
      rx_pck.cmd = CMD_INVALID;
      rx_pck.dataLen = 0;

      decodePacket(pck_rx_buffer, pck_rx_index + 1, &rx_pck);

      if (!connected && rx_pck.cmd == CMD_CONNECT)
      {
        connected = true;
        iniMoveo();

        tx_pck.cmd = CMD_CONNECT;
        tx_pck.dataLen = 0;
        writePacket(&tx_pck);
      }
      else if (connected)
      {
        switch (rx_pck.cmd)
        {
        case CMD_CONNECT:
        {
          tx_pck.cmd = CMD_CONNECT;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_SET_MOTOR_PARAMS:
        {

          if (rx_pck.dataLen == (6 /* Define the number of motors*/ * sizeof(float)) * 2)
          {

            memcpy(&motor_maxVel, rx_pck.data, 6 /* Define the number of motors*/ * sizeof(float));
            memcpy(&move_Accel, rx_pck.data + 6 /* Define the number of motors*/ * sizeof(float), 6 /* Define the number of motors*/ * sizeof(float));

            setMotorParams(motor_maxVel, move_Accel);

            tx_pck.cmd = CMD_SET_MOTOR_PARAMS;
            tx_pck.dataLen = 0;
          }
          else
          {
            tx_pck.cmd = CMD_INVALID;
            tx_pck.dataLen = 0;
          }
        }
        break;

        case CMD_SET_MOTOR_RANGES:
        {

          if (rx_pck.dataLen == 6 /* Define the number of motors*/ * sizeof(int16_t) * 2)
          {

            for (int i = 0; i < 6 /* Define the number of motors*/; ++i)
            {
              // Extract lower limit
              motor_ranges[i * 2] = (int16_t)((uint16_t)rx_pck.data[i * 4] << 8 | (uint16_t)rx_pck.data[i * 4 + 1]);
              // Extract upper limit
              motor_ranges[i * 2 + 1] = (int16_t)((uint16_t)rx_pck.data[i * 4 + 2] << 8 | (uint16_t)rx_pck.data[i * 4 + 3]);
            }

            setMotorLimits(motor_ranges);

            tx_pck.cmd = CMD_SET_MOTOR_RANGES;
            tx_pck.dataLen = 0;
          }
          else
          {
            tx_pck.cmd = CMD_INVALID;
            tx_pck.dataLen = 0;
          }
        }
        break;

        case CMD_ENABLE:
        {
          enableMoveo();
          tx_pck.cmd = CMD_ENABLE;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_DISABLE:
        {
          disableMoveo();
          tx_pck.cmd = CMD_DISABLE;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_STOP:
        {
          stopMoveo();
          motion_type = MotionType::STILL;

          tx_pck.cmd = CMD_STOP;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_SET_TARGET_POSITION:
        {
          if (rx_pck.dataLen == 6 /* Define the number of motors*/ * sizeof(int16_t))
          {
            for (int i = 0; i < 6 /* Define the number of motors*/; ++i)
            {
              // Extract target position for each motor
              motor_pos[i] = (int16_t)((uint16_t)rx_pck.data[i * 2] << 8 | (uint16_t)rx_pck.data[i * 2 + 1]);
            }

            setTargetPosition(motor_pos);

            tx_pck.cmd = CMD_SET_TARGET_POSITION;
            tx_pck.dataLen = 0;
          }
          else
          {
            tx_pck.cmd = CMD_INVALID;
            tx_pck.dataLen = 0;
          }
        }
        break;

        case CMD_GET_CURRENT_POSITION:
        {
          tx_pck.cmd = CMD_GET_CURRENT_POSITION;
          tx_pck.dataLen = sizeof(int16_t) * 6 /* Define the number of motors*/;

          reportCurrentPositions(motor_pos);

          // Extract Current motor position
          for (int i = 0; i < 6 /* Define the number of motors*/; ++i)
          {
            tx_pck.data[2 * i] = (uint8_t)(motor_pos[i] >> 8);
            tx_pck.data[2 * i + 1] = (uint8_t)(motor_pos[i] & 0xFF);
          }
        }
        break;

        case CMD_OPEN_GRIPPER:
        {
          openGripper();
          tx_pck.cmd = CMD_OPEN_GRIPPER;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_CLOSE_GRIPPER:
        {
          closeGripper();
          tx_pck.cmd = CMD_CLOSE_GRIPPER;
          tx_pck.dataLen = 0;
        }
        break;

        case CMD_MOVE_TO_POSITION:
        {
          if (rx_pck.dataLen == (6 /* Define the number of motors*/ * sizeof(float)) * 2)
          {
            memcpy(&move_maxVel, rx_pck.data, 6 /* Define the number of motors*/ * sizeof(float));
            memcpy(&move_Accel, rx_pck.data + 6 /* Define the number of motors*/ * sizeof(float), 6 /* Define the number of motors*/ * sizeof(float));

            setVelAccel(move_maxVel, move_Accel);

            motion_type = MotionType::JOINT_INTERP;

            tx_pck.cmd = CMD_MOVE_TO_POSITION;
            tx_pck.dataLen = 0;
          }
          else
          {
            tx_pck.cmd = CMD_INVALID;
            tx_pck.dataLen = 0;
          }
        }
        break;

        case CMD_TRAJ_TO_POSITION:
        {
          if (rx_pck.dataLen == 6 /* Define the number of motors*/ * sizeof(float))
          {

            memcpy(&traj_Vel, rx_pck.data, 6 /* Define the number of motors*/ * sizeof(float));

            setSpeed(traj_Vel);

            tx_pck.cmd = CMD_TRAJ_TO_POSITION;
            tx_pck.dataLen = 0;

            motion_type = MotionType::SPACE_INTERP;
          }
          else
          {
            tx_pck.cmd = CMD_INVALID;
            tx_pck.dataLen = 0;
          }
        }
        break;
        }

        writePacket(&tx_pck);
      }
      else // Error
      {
      }
      pck_rx_index = 0;
    }
    else
    {
      ++pck_rx_index;
      if (pck_rx_index >= 128 /*??*/)
        pck_rx_index = 0;
    }
  }

  // Change motion_type only if all motors have already reached final position.
  if (!checkMotorsRunning())
  {
    motion_type = MotionType::STILL;
    still_first_time = true;
  }

  // Moves the robot depending on the type of movement.
  if (motion_type == MotionType::JOINT_INTERP)
  {
    moveToPosition();

  }

  else if (motion_type == MotionType::SPACE_INTERP)
  {
    trajToPosition();

  }

  else if (motion_type == MotionType::STILL)
  {
    if(still_first_time == true){
      setMotorParams(motor_maxVel,motor_maxAccel);
      still_first_time = false;
    }
  }
}
