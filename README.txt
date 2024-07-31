
# Robotics Toolbox for Arduino-based Manipulators

This project offers an easy-to-understand toolbox designed for robotics enthusiasts and students to grasp fundamental concepts such as Forward Kinematics, Denavit-Hartenberg (DH) Tables, Jacobians, and Inverse Kinematics. The toolbox is specifically tailored for the Moveo3D robotic arm but can be easily adapted to other Arduino-based manipulators with minimal modifications.

## Project Overview

The repository is divided into three main components:

### 1. moveo_driver
The `moveo_driver` component handles the communication protocol, preparing data for transmission via the serial port. It utilizes the Consistent Overhead Byte Stuffing (COBS) protocol for efficient data encoding, creating packets that are easily interpreted by the microcontroller. This part is implemented in C++ and bound to the software block using Pybind11.

### 2. moveo_firmware
The `moveo_firmware` component defines the actions the robot must perform upon receiving specific packets. It directly controls the motors (stepper motors for joints and a servo motor for the gripper), ensuring the received data translates into the desired movements.

### 3. moveo_software
The `moveo_software` component is responsible for calculating and creating the data packets. It allows users to define the DH Table of the robot and specify the motor distribution for each joint.

## Compatibility
This repository currently supports only the Moveo3D robotic arm with a specific configuration. However, the code can be easily adapted to work with other Arduino-based robotic manipulators by making minor adjustments.

## Installation Requirements

To run this project, you will need to install the following packages:

| Package                  | Version  |
|--------------------------|----------|
| ansitable                | 0.10.0   |
| cfgv                     | 3.4.0    |
| colored                  | 2.2.4    |
| contourpy                | 1.2.1    |
| cycler                   | 0.12.1   |
| distlib                  | 0.3.8    |
| filelock                 | 3.15.4   |
| fonttools                | 4.53.1   |
| identify                 | 2.6.0    |
| kiwisolver               | 1.4.5    |
| matplotlib               | 3.8.3    |
| matplotlib-inline        | 0.1.6    |
| moveo_driver             | 1.0      |
| mpmath                   | 1.3.0    |
| nodeenv                  | 1.9.1    |
| numpy                    | 1.26.4   |
| packaging                | 24.1     |
| pgraph-python            | 0.6.2    |
| pillow                   | 10.4.0   |
| pip                      | 24.0     |
| platformdirs             | 4.2.2    |
| pre-commit               | 3.7.1    |
| progress                 | 1.6      |
| pybind11                 | 2.13.1   |
| pyparsing                | 3.1.2    |
| python-dateutil          | 2.9.0.post0 |
| PyYAML                   | 6.0.1    |
| roboticstoolbox-python   | 1.1.1    |
| rtb-data                 | 1.0.1    |
| scipy                    | 1.12.0   |
| six                      | 1.16.0   |
| spatialgeometry          | 1.1.0    |
| spatialmath-python       | 1.1.10   |
| swift-sim                | 1.1.0    |
| sympy                    | 1.12     |
| traitlets                | 5.14.3   |
| typing_extensions        | 4.12.2   |
| virtualenv               | 20.26.3  |
| websockets               | 12.0     |

To install the necessary packages, you can use the following command:
```bash
pip install -r requirements.txt
```

For `moveo_driver`, use the following command:
```bash
pip install ./moveo_driver/
```

## Getting Started

1. Clone this repository to your local machine.
2. Install the required packages as described above.
3. Follow the documentation to configure the DH Table and motor settings for your specific robotic manipulator.

## Contributions and Support

We welcome contributions to enhance this project and expand its compatibility with other robotic arms. Please submit your pull requests and issues on our GitHub repository.

For any questions or support, feel free to reach out via the issue tracker.
