#include "moveo_functions.h"

// Define pin numbers for enabling/disabling the motors
const int ENB_pins[NUM_MOTORS] = {48, 40, 32, 22, 49, 41};

// Define the array to store lower and upper limits for each motor
int16_t motorLimits[NUM_MOTORS][2];

// Define the pin to which the servo is connected
const int servoPin = 9;

// Create a Servo object
Servo gripper;

// Define the angles for opening and closing the gripper
const int openAngle = 90;
const int closeAngle = 170;

AccelStepper motors[NUM_MOTORS] = {
    AccelStepper(1, 52, 50), // J1
    AccelStepper(1, 44, 42), // J21
    AccelStepper(1, 36, 34), // J22
    AccelStepper(1, 24, 26), // J3
    AccelStepper(1, 53, 51), // J4
    AccelStepper(1, 45, 43)  // J5
};

// Inatialize the moveo arm.
void iniMoveo()
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        motors[i].setMaxSpeed(MAX_SPEED);
        motors[i].setAcceleration(MAX_ACCEL);
        pinMode(ENB_pins[i], OUTPUT);
    }

    gripper.attach(servoPin);
}

// Sets maxVelocity and Acceleration for each motor.
void setMotorParams(float maxVelocity[], float Acceleration[])
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        motors[i].setMaxSpeed(maxVelocity[i]);
        motors[i].setAcceleration(Acceleration[i]);
    }
}

// Function to set the limits for each motor
void setMotorLimits(const int16_t limits[])
{
    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        motorLimits[i][0] = limits[2 * i];     // Lower limit
        motorLimits[i][1] = limits[2 * i + 1]; // Upper limit
    }
}

// Function to check and enforce motor limits, stopping motors if limits are exceeded
void checkAndEnforceLimits()
{
    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        int16_t currentPosition = motors[i].currentPosition();
        if (currentPosition < motorLimits[i][0] || currentPosition > motorLimits[i][1])
        {
            motors[i].stop(); // Stop the motor if it exceeds limits
        }
    }
}

// Gives Power to the motors
void enableMoveo()
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        digitalWrite(ENB_pins[i], LOW);
    }
}

// Remove Power of the motors
void disableMoveo()
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        digitalWrite(ENB_pins[i], HIGH);
    }
}

// Moves the servo to the open angle
void openGripper()
{

    gripper.write(openAngle);
}

// Moves the servo to the closed angle
void closeGripper()
{

    gripper.write(closeAngle);
}

// Stops the movement
void stopMoveo()
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {

        motors[i].stop();
    }
}

// Reports the currents positions of each joint.
void reportCurrentPositions(int16_t positions[])
{

    // Get and store the current positions of each motor
    for (int i = 0; i < NUM_MOTORS; ++i)
    {

        positions[i] = motors[i].targetPosition();
    }
}

// Function to set the target position for each motor, ensuring it does not exceed limits
void setTargetPosition(const int16_t positions[])
{
    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        // Clamp the input position within the defined limits
        int16_t clampedPosition = positions[i];

        if (positions[i] < motorLimits[i][0])
        {
            clampedPosition = motorLimits[i][0];
        }
        else if (positions[i] > motorLimits[i][1])
        {
            clampedPosition = motorLimits[i][1];
        }

        motors[i].moveTo(clampedPosition);
    }
}

// Sets maxVelocity and Acceleration for each motor.
void setVelAccel(float maxVelocity[], float Acceleration[])
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        motors[i].setMaxSpeed(maxVelocity[i]);
        motors[i].setAcceleration(Acceleration[i]);
    }
}

// Sets the speed at wich each motor has to move
void setSpeed(const float velocities[])
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {

        motors[i].setSpeed(velocities[i]);
    }
}

// Moves each motor if a step is due. Moves to the objective position without following any trajectory
void moveToPosition()
{

    for (int i = 0; i < NUM_MOTORS; ++i)
    {

        motors[i].run();
        checkAndEnforceLimits();
    }
}

void trajToPosition()
{

    // Move each motor to its final position
    for (int i = 0; i < NUM_MOTORS; i++)
    {

        motors[i].runSpeed();
        checkAndEnforceLimits();
    }
}

bool checkMotorsRunning()
{

    bool anyMotorRunning = false;

    for (int i = 0; i < NUM_MOTORS; i++)
    {

        long distance = motors[i].distanceToGo();

        if (abs(distance) > STEP_TOLERANCE)
        {

            // Any motor not within tolerance is still running
            anyMotorRunning = true;
        }
    }
    return anyMotorRunning; // Return true if any motor is still running
}
