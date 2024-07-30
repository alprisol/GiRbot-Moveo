#line 1 "/home/cirs/moveo/moveo_firmware/moveo_functions.h"
#ifndef MOVEO_FUNCTIONS_H
#define MOVEO_FUNCTIONS_H

#include <Vector.h>
#include <AccelStepper.h>
#include <Servo.h>
#include <Arduino.h>

#define NUM_MOTORS 6 // Define the number of motors

// Define pin numbers for enabling/disabling the motors
extern const int ENB_pins[NUM_MOTORS]; // Defined in the implementation file

// Define maximum speed and acceleration for the motors
#define MAX_SPEED 3000
#define MAX_ACCEL 3000

// Tolerance for the target position
#define STEP_TOLERANCE 5

// Define the pin to which the servo is connected
extern const int servoPin;

// Create a Servo object
extern Servo gripper;

// Define the angles for opening and closing the gripper
extern const int openAngle;
extern const int closeAngle;

extern AccelStepper motors[NUM_MOTORS];

void iniMoveo();
void setMotorParams(float maxVelocity[], float Acceleration[]);
void setMotorLimits(const int16_t limits[]);
void enableMoveo();
void disableMoveo();
void openGripper();
void closeGripper();
void stopMoveo();
void reportCurrentPositions(int16_t positions[]);
void setTargetPosition(const int16_t positions[]);
void setVelAccel(float maxVelocity[], float Acceleration[]);
void setSpeed(const float velocities[]);
void moveToPosition();
void trajToPosition();
bool checkMotorsRunning();

#endif