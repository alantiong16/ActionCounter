import sys
import numpy as np
import open3d as o3d
import cv2
# sys.path.append('C:\\Users\\cic\\Documents\\pyKinectAzure')
# from pykinect_azure.pykinect_azure import *
import pykinect_azure as pykinect
import matplotlib.pyplot as plt  
import time 

KINECT_NODES = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}
def calculate_right_elbow_angle(rshoulder, relbow, rwrist):
    v1 = np.array(relbow) - np.array(rshoulder)
    v2 = np.array(rwrist) - np.array(relbow)

    # dot_product = np.dot(v1, v2)
    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
 
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate angle in radians and convert to degrees
    relbow_angle_deg = np.degrees(np.arccos(cos_theta))
    return relbow_angle_deg

def calculate_right_knee_angle(rankle, rknee, rhip):
    v1 = np.array(rknee) - np.array(rhip)
    v2 = np.array(rankle) - np.array(rknee)

    # dot_product = np.dot(v1, v2)
    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
 
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate angle in radians and convert to degrees
    rknee_angle_deg = np.degrees(np.arccos(cos_theta))

    return rknee_angle_deg


def calculate_right_shoulder_angle(relbow, rshoulder, rclavicle):
    v1 = np.array(rclavicle) - np.array(rshoulder)
    v2 = np.array(rshoulder) - np.array(relbow)

    # Calculate the signed angle using atan2
    rshoulder_angle_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    # Convert the angle to degrees
    rshoulder_angle_deg = np.degrees(rshoulder_angle_rad)

    return rshoulder_angle_deg

def calculate_pelvis_angle(rknee, pelvis, lknee):
    v1 = np.array(pelvis) - np.array(rknee)
    v2 = np.array(pelvis) - np.array(lknee)

    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate angle in radians and convert to degrees
    pelvis_angle_deg = np.degrees(np.arccos(cos_theta))

    return pelvis_angle_deg

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

class KinectLib:
    def __init__(self) -> None:
        
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
       
        self.device = pykinect.start_device(config=device_config)

        self.bodyTracker = pykinect.start_body_tracker()

    def get_skel_pos(self):
        self.capture = self.device.update()

		# Get body tracker frame
        self.body_frame = self.bodyTracker.update()

        kinect_skel = {}
        # if self.body_frame.get_num_bodies():
        
        #     body3d = self.body_frame.get_body(0)
        #     body3d_joints = body3d.joints
        #     # print(body3d_joints)
        #     xyz = []
        #     # quat = []
        #     for joint in body3d_joints:
        #         xyz.append(joint.get_xyz())
                
        #     # xyz = X_180_FLIP.transforms() @ np.array(xyz).T
        #     # xyz = xyz.T
        #     for NODE in KINECT_NODES:
        #         kinect_skel[NODE] = xyz[KINECT_NODES[NODE]].tolist()

        #     return kinect_skel
        if self.body_frame.get_num_bodies():
            body3d = self.body_frame.get_body(0)
            body3d_joints = body3d.joints

            for NODE in KINECT_NODES:
                kinect_skel[NODE] = body3d_joints[KINECT_NODES[NODE]].get_xyz().tolist()

        return kinect_skel if kinect_skel else None
       

    def stop_device(self):
        self.device.close()

    
        
if __name__ == '__main__':
    kin = KinectLib()

    lelbow_angles = []
    relbow_angles = []
    lknee_angles = []
    rknee_angles = []
    rshoulder_angles = []
    pelvis_angles = []
    frame_numbers = []
    squatflag1=False
    squatflag2=False
    jjflag1=False
    jjflag2=False
    puflag1=False
    puflag2=False

    try:
        frame_number=0
        push_up_count = 0
        squat_count = 0
        jumping_jack_count = 0
        last_time = time.time()
        update_interval = 5
        fig, (ax_elbow, ax_knee, ax_shoulder) = plt.subplots(3, 1, sharex=True)

        line_elbow, = ax_elbow.plot([], [], label='Elbow Angles')
        line_knee, = ax_knee.plot([], [], label='Knee Angles')

        ax_elbow.set_ylabel('Elbow Angle (degrees)')
        ax_elbow.set_title('Elbow Angle vs Frame Number')
        ax_elbow.legend()

        ax_knee.set_xlabel('Frame Number')
        ax_knee.set_ylabel('Knee Angle (degrees)')
        ax_knee.set_title('Knee Angle vs Frame Number')
        ax_knee.legend()

        ax_knee.set_xlabel('Frame Number')
        ax_knee.set_ylabel('Shoulder Angle (degrees)')
        ax_knee.set_title('Shoulder Angle vs Frame Number')
        ax_knee.legend()

        while True:
            # print(kin.get_skel_quat())
            # print(kin.get_skel_pos())
            
            
            pos = kin.get_skel_pos()
            if pos is not None:
                capture = kin.capture

                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - last_time)

                body_frame=kin.body_frame
                ret_depth, depth_color_image = capture.get_colored_depth_image()

                # Get the colored body segmentation
                ret_color, body_image_color = body_frame.get_segmentation_image()

                if not ret_depth or not ret_color:
                    continue
                
                # Combine both images
                combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

                # Draw the skeletons
                combined_image = body_frame.draw_bodies(combined_image)
                # Overlay FPS on the image
                cv2.putText(combined_image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Overlay body segmentation on depth image
                

                shoulder_right_coords = pos['SHOULDER_RIGHT']
                elbow_right_coords = pos['ELBOW_RIGHT']
                wrist_right_coords = pos['WRIST_RIGHT']

                ankle_right_coords = pos['ANKLE_RIGHT']
                knee_right_coords = pos['KNEE_RIGHT']
                hip_right_coords = pos['HIP_RIGHT']

                elbow_right_coords = pos['ELBOW_RIGHT']
                shoulder_right_coords = pos['SHOULDER_RIGHT']
                clavicle_right_coords = pos['CLAVICLE_RIGHT']
                
                knee_right_coords = pos['KNEE_RIGHT']
                knee_left_coords = pos['KNEE_LEFT']
                pelvis_coords = pos['PELVIS']

                with Timer('counting'):
                    elbow_right_angle = calculate_right_elbow_angle(shoulder_right_coords, elbow_right_coords, wrist_right_coords)
                    if elbow_right_angle > 90:
                        puflag1 = True
                    if elbow_right_angle < 10 and puflag1 == True:
                        puflag2 = True
                    if puflag1 == True and puflag2 == True:
                        push_up_count+=1
                        puflag1=False
                        puflag2=False

                    print(f"Elbow Angle: {elbow_right_angle} degrees")
                    relbow_angles.append(elbow_right_angle)
                    cv2.putText(combined_image, f'push up: {int(push_up_count)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    knee_right_angle = calculate_right_knee_angle(ankle_right_coords, knee_right_coords, hip_right_coords)
                    if knee_right_angle > 90:
                        squatflag1 = True
                    if knee_right_angle < 70  and squatflag1==True:
                        squatflag2 = True
                    if squatflag1==True and squatflag2==True:
                        squat_count+=1
                        squatflag1=False
                        squatflag2=False
                    # print(f"Count: {squat_count} times")

                    cv2.putText(combined_image, f'squat: {int(squat_count)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    rknee_angles.append(knee_right_angle)
                
                    shoulder_right_angle = calculate_right_shoulder_angle(elbow_right_coords, shoulder_right_coords, clavicle_right_coords)
                    pelvis_angle = calculate_pelvis_angle(knee_right_coords, pelvis_coords, knee_left_coords)
                    if shoulder_right_angle > 35 and pelvis_angle > 35:
                        jjflag1 = True
                    if shoulder_right_angle < -35 and pelvis_angle < 25 and jjflag1 == True:
                        jjflag2 = True
                    if jjflag1 == True and jjflag2 == True:
                        jumping_jack_count+=1
                        jjflag1 = False
                        jjflag2 = False

                    # print(f"jumping jack count: {jumping_jack_count} degrees")
                    rshoulder_angles.append(shoulder_right_angle)
                    cv2.putText(combined_image, f'jumping_jack: {int(jumping_jack_count)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


                    
                    # if pelvis_angle > 45:
                    #     flag1 = True
                    # if pelvis_angle < 20 and flag1 == True:
                    #     flag2 = True
                    # if flag1 == True and flag2 == True:
                    #     jumping_jack_count+=1
                    #     flag1=False
                    #     flag2=False

                    # print(f"jumping jack count: {jumping_jack_count} degrees")
                    # rshoulder_angles.append(pelvis_angle)
                    # cv2.putText(combined_image, f'jumping jack count: {int(pelvis_angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    frame_numbers.append(frame_number) 
                    # Update last_time for the next iteration
                    last_time = current_time
                    cv2.imshow('Depth image with skeleton',combined_image)
                with Timer('plotting'):
                    if frame_number % update_interval == 0:
                       # Update subplot lines
                        line_elbow.set_xdata(frame_numbers)
                        line_elbow.set_ydata(rknee_angles)

                        line_knee.set_xdata(frame_numbers)
                        line_knee.set_ydata(relbow_angles)

                        line_knee.set_xdata(frame_numbers)
                        line_knee.set_ydata(rshoulder_angles)

                        # Set plot limits (optional)
                        ax_elbow.set_xlim(min(frame_numbers), max(frame_numbers))
                        ax_elbow.set_ylim(0, 180)



                        # Redraw the plot
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    frame_number += 1
                    
                    plt.pause(0.01)  
                
                # Press q key to stop
                if cv2.waitKey(1) == ord('q'):  
                    break
            else:
                print("No human detected.")

    finally:

        kin.device.close()
