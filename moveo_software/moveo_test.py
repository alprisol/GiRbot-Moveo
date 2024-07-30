from Moveo3D import Moveo
import math


Moveo.cmd_Init()

Moveo.cmd_MoveJ(trgtQ=[0, 0, 0, 0, 0], maxVel=0.5, Accel=0.1)
Moveo.cmd_WaitEndMove()
Moveo.cmd_closeGripper()

Moveo.cmd_MoveL(
    trgtQ=[math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0], cartVel=0.006, cartAccel=0.1, mask = [1,1,1,1,0,0]
)
Moveo.cmd_WaitEndMove()
Moveo.cmd_getJointValues()

Moveo.cmd_MoveJ(trgtQ=[math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0], maxVel=1, Accel=0.5)
Moveo.cmd_WaitEndMove()

Moveo.cmd_openGripper()

Moveo.cmd_MoveJ(trgtQ=Moveo.home_q,maxVel = 1, Accel = 0.5)
Moveo.cmd_WaitEndMove()
