from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse
import imutils
import math                           
from imutils.video import VideoStream    
from imutils.video import FPS            

from simple_pid import PID

from cv2 import aruco

# Frames per second of the pygame window display
FPS = 25
dimensions = (960, 720)
xoff = 5
yoff = 5
zoff = 5
vTarget = []
inis = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960, 720))

""" Camera calibration parameters
You should calibrate your camera before using this code."""
with open ("camera_calibration.npy","rb") as f:
    mtx=np.load(f)
    dist=np.load(f)

markerSize = 15 #cm

aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) # Use 6x6 dictionary to find markers

parameters =  aruco.DetectorParameters_create()


def isRotationMatrix(R):
    """Checks if a matrix is a valid rotation matrix."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

starttime = 0
class FrontEnd(object):
    
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        self.send_rc_control = False


        self.pid_yaw = PID(0.20, 0.00005, 0.01,setpoint=0,output_limits=(-100,100))
        self.pid_throttle = PID(0.25, 0.00001, 0.01,setpoint=0,output_limits=(-100,50))
        self.pid_pitch = PID(0.25, 0.0002, 0.01,setpoint=0,output_limits=(-20,20))
        self.pid_roll = PID(0.35, 0.00005, 0.01,setpoint=0,output_limits=(-70,70))
        self.takeoff=False



    def run(self):

        if not self.tello.connect():
            self.tello.connect()
            print("Tello connected")
           

#        if not self.tello.set_speed(self.speed):
#            print("Not set speed to lowest possible")
#            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
           

        if not self.tello.streamon():
            print("Could not start video stream")
        

        frame_read = self.tello.get_frame_read()

        cv2.namedWindow('Tello Tracking...')


        should_stop = False
        imgCount = 0
        OVERRIDE = False
        inis = False
    
        countVar = 0

        global xoff
        global yoff
        global zoff
        global vTarget
        global starttime

        self.tello.get_battery()
      
        
        while not should_stop:
            self.update()
            
            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Frame",frame)
            frameRet = frame_read.frame



            vid = self.tello.get_video_capture()

            
            frame = np.rot90(frame)
            imgCount+=1

            time.sleep(1 / FPS)

            # Listen for key presses
            k = cv2.waitKey(20)
            if not self.takeoff:
                print("take-off succesfull")
                self.tello.takeoff()
                self.takeoff=True
                self.send_rc_control=True
                
                


            endtime = time.time()
            current_time = endtime - starttime
            if inis == True and current_time > 5 and countVar<2:
                self.tello.land()
                countVar = countVar + 1


            # Quit the software
            if k == 27:
                should_stop = True
                break

     
    # if the `q` key was pressed, break from the loop
            elif k == ord("q"):
                    break

            if self.send_rc_control and not OVERRIDE:
                 

                gray = cv2.cvtColor(frameRet, cv2.COLOR_BGR2GRAY)
                """ Detect the markers in the image"""
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


                if np.all(ids != None):
                   

                    corner = corners[0][0]
                    """ Find the center of the marker"""
                    m = int((corner[0][0]+corner[1][0]+corner[2][0]+corner[3][0])/4)
                    n = int((corner[0][1]+corner[1][1]+corner[2][1]+corner[3][1])/4)
                    orta = int((corner[0][0]+corner[3][0])/2)


                    ret = aruco.estimatePoseSingleMarkers(corners, markerSize, mtx, dist)
                    rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
                    """ Draw the axis of the marker"""
                    aruco.drawDetectedMarkers(frameRet.copy(), corners, ids)
                    aruco.drawAxis(frameRet, mtx, dist, rvec, tvec, 10)
                    """ Find the distance between the center of the marker and the center of the image"""
                    str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
                    cv2.putText(frameRet, str_position, (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
                    R_tc    = R_ct.T
                    """ Find the rotation matrix between the camera and the marker"""
                    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)

                    str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker), math.degrees(yaw_marker))
                    cv2.putText(frameRet, str_attitude, (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    """ Find the position of the camera in the marker coordinate system"""
                    pos_camera = -R_tc*np.matrix(tvec).T

                    str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
                    cv2.putText(frameRet, str_position, (0, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
                    roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
                    str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera), math.degrees(yaw_camera))
                    cv2.putText(frameRet, str_attitude, (0, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("Frame2",frameRet)
                    out.write(frameRet)
                    targ_cord_x = m
                    targ_cord_y = n
                    """ Find the position of the marker in the drone coordinate system"""
                    xoff = int(targ_cord_x - 480)
                    yoff = int(540-targ_cord_y)
                    zoff = int(90-tvec[2]) 
                    roff = int(95-math.degrees(yaw_marker))
                    vTarget = np.array((xoff,yoff,zoff,roff))
                    """ Control the velocity of the drone USING PID"""
                    self.yaw_velocity = int(-self.pid_yaw(xoff))

                    self.up_down_velocity = int(-self.pid_throttle(yoff))

                    self.for_back_velocity = int(self.pid_pitch(zoff))

                    self.left_right_velocity = int(self.pid_roll(roff))




                    if -15<xoff<15 and -15<yoff<15 and -45<zoff<45 and roff<15:
                        """ If the drone is close to the marker, it will land"""
                        landingSpeed = int((0.8883*tvec[2])-3.4264)
                        print("sending this",landingSpeed)
                    
                        self.tello.move_forward(landingSpeed)
                        inis = True
                        countVar = 1
                        starttime = time.time()
                        self.send_rc_control = False


                else:

                    self.yaw_velocity = 0

                    self.up_down_velocity = 0

                    self.for_back_velocity = 0
 
                    self.left_right_velocity = 0



            cv2.putText(frameRet,str(vTarget),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                
            # Draw the center of screen circle, this is what the drone tries to match with the target coords
            cv2.circle(frameRet, (480, 540), 10, (0,0,255), 2)
            cv2.imshow('Tello Tracking...',frameRet)


            # Display the resulting frame
        cv2.imshow('Tello Tracking...',frameRet)
       
        key = cv2.waitKey(1) & 0xFF
        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()
        
        # Call it always before finishing. I deallocate resources.
        self.tello.end()


    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
           
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

def main():
    frontend = FrontEnd()
    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
