import sys
import numpy as np
import open3d as o3d
import cv2
# sys.path.append('C:\\Users\\cic\\Documents\\pyKinectAzure')
# from pykinect_azure.pykinect_azure import *
import pykinect_azure as pykinect
import matplotlib.pyplot as plt  

KINECT_NODES = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}
def calculate_elbow_angle(shoulder, elbow, wrist):
    v1 = np.array(elbow) - np.array(shoulder)
    v2 = np.array(elbow) - np.array(wrist)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
 
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Ensure cos_theta is within valid range [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(cos_theta)

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg
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
        if self.body_frame.get_num_bodies():
            body3d = self.body_frame.get_body(0)
            body3d_joints = body3d.joints
            # print(body3d_joints)
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                
            # xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            # xyz = xyz.T
            for NODE in KINECT_NODES:
                kinect_skel[NODE] = xyz[KINECT_NODES[NODE]].tolist()

            return kinect_skel
       

    def stop_device(self):
        self.device.close()

    
        
if __name__ == '__main__':
    kin = KinectLib()
    try:
        while True:
            # print(kin.get_skel_quat())
            # print(kin.get_skel_pos())
            pos = kin.get_skel_pos()
            # print(pos['PELVIS'])
            # if isinstance(pos, dict):
            #     line_set = o3d.geometry.LineSet()

            #     # Rest of the code...
            # else:
            #     print("No bodies detected.")

            # line_set = o3d.geometry.LineSet()

            # points = []
            # for joint_name, coordinates in pos.items():
            #     x, y, z = coordinates
            #     points.append([x, y, z])

            # points_array = np.array(points)

            # line_set.points = o3d.utility.Vector3dVector(points_array)

            # connections = [
            #     ('PELVIS', 'SPINE_NAVEL'),
            #     ('SPINE_NAVEL', 'SPINE_CHEST'),
            #     ('SPINE_CHEST', 'NECK'),
            #     ('NECK', 'HEAD'),
            #     ('HEAD', 'NOSE'),
            #     ('NOSE', 'EYE_RIGHT'),
            #     ('EYE_RIGHT', 'EAR_RIGHT'),
            #     ('NOSE', 'EYE_LEFT'),
            #     ('EYE_RIGHT', 'EAR_LEFT'),
            #     ('SPINE_CHEST', 'CLAVICLE_RIGHT'),
            #     ('CLAVICLE_RIGHT', 'SHOULDER_RIGHT'),
            #     ('SHOULDER_RIGHT', 'ELBOW_RIGHT'),
            #     ('ELBOW_RIGHT', 'WRIST_RIGHT'),
            #     ('WRIST_RIGHT', 'HAND_RIGHT'),
            #     ('WRIST_RIGHT', 'THUMB_RIGHT'),
            #     ('HAND_RIGHT', 'HANDTIP_RIGHT'),
            #     ('SPINE_CHEST', 'CLAVICLE_LEFT'),
            #     ('CLAVICLE_LEFT', 'SHOULDER_LEFT'),
            #     ('SHOULDER_LEFT', 'ELBOW_LEFT'),
            #     ('ELBOW_LEFT', 'WRIST_LEFT'),
            #     ('WRIST_LEFT', 'HAND_LEFT'),
            #     ('WRIST_LEFT', 'THUMB_LEFT'),
            #     ('HAND_LEFT', 'HANDTIP_LEFT'),
            #     ('PELVIS', 'HIP_RIGHT'),
            #     ('HIP_RIGHT', 'KNEE_RIGHT'),
            #     ('KNEE_RIGHT', 'ANKLE_RIGHT'),
            #     ('ANKLE_RIGHT', 'FOOT_RIGHT'),
            #     ('PELVIS', 'HIP_LEFT'),
            #     ('HIP_LEFT', 'KNEE_LEFT'),
            #     ('KNEE_LEFT', 'ANKLE_LEFT'),
            #     ('ANKLE_LEFT', 'FOOT_LEFT'),
            # ]

            # lines = []
            # for start_joint, end_joint in connections:
            #     start_index = KINECT_NODES[start_joint]
            #     end_index = KINECT_NODES[end_joint]
            #     lines.append([st art_index, end_index])

            # line_set.lines = o3d.utility.Vector2iVector(lines)

            # o3d.visualization.draw_geometries([line_set])

            capture = kin.capture
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

            # Overlay body segmentation on depth image
            cv2.imshow('Depth image with skeleton',combined_image)

            shoulder_coords = pos['SHOULDER_RIGHT']
            elbow_coords = pos['ELBOW_RIGHT']
            wrist_coords = pos['WRIST_RIGHT']

            elbow_angle = calculate_elbow_angle(shoulder_coords, elbow_coords, wrist_coords)
            print(f"Elbow Angle: {elbow_angle} degrees")

            # Append the current elbow angle to the list

            # Plot the elbow angles over time
            plt.plot(frame_numbers, elbow_angle, label='Elbow Angle')
            plt.xlabel('Frame')
            plt.ylabel('Elbow Angle (degrees)')
            plt.title('Elbow Angle Over Time')
            plt.legend()

            # Display the plot
            plt.pause(0.01)
            
            frame_numbers += 1
            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):  
                break
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully

    finally:
        kin.device.close()
        plt.show()  # Show the final plot after closing the Kinect device