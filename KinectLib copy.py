import sys
import numpy as np
# sys.path.append('C:\\Users\\cic\\Documents\\pyKinectAzure')
# from pykinect_azure.pykinect_azure import *
import pykinect_azure as pykinect
# from plot3dUtils import Open3dVisualizer, Open3dMeshVisualizer
# print (pykinect.__file__)
sys.path.append(r'../bvh_utils')
# sys.path.append(r'../STAR')
# sys.path.append(r'C:/Users/cic/Documents/human_body_prior/src')
# from human_body_prior.models.ik_engine import IK_Engine
# from human_body_prior.body_model.body_model import BodyModel
# from star.pytorch.star import STAR
from Quaternions_old import Quaternions
from pytorch3d import transforms
import torch
import json
from QuaternionLookRotation import LookRotation


KINECT_NODES = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}

zpositive = np.array([0,0,1]) # forward
xpositive = np.array([1,0,0]) # right
ypositive = np.array([0,1,0]) # up

X_180_FLIP = Quaternions(np.array([0,1,0,0]))
Y_180_FLIP = Quaternions(np.array([0,0,1,0]))
Z_180_FLIP = Quaternions(np.array([0,0,0,1]))

# hipleftBasis = LookRotation(forward=ypositive, up=xpositive)
# spineHipBasis = LookRotation(forward=ypositive, up=xpositive)
hipleftBasis = LookRotation(forward=-xpositive, up=zpositive)
# spineHipBasis = LookRotation(forward=-zpositive, up=-ypositive)
spineHipBasis = LookRotation(forward=xpositive, up=zpositive)
# print(spineHipBasis)
hiprightBasis = LookRotation(-xpositive, -zpositive)

# leftArmBasis = LookRotation(zpositive, xpositive)
leftArmBasis = LookRotation(-ypositive, zpositive)

rightArmBasis = LookRotation(-xpositive, zpositive)

leftHandBasis = LookRotation(-zpositive, -ypositive)
rightHandBasis = Quaternions(np.array([1,0,0,0]))
leftFootBasis = LookRotation(xpositive, ypositive)
rightFootBasis = LookRotation(xpositive, -ypositive)


class KinectLib:
    def __init__(self) -> None:
        
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
       
        self.device = pykinect.start_device(config=device_config)

        self.bodyTracker = pykinect.start_body_tracker()


    def get_skel_pos(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                # quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break

            # xyz = np.array(xyz)

            for NODE in KINECT_NODES:
                kinect_skel[NODE] = xyz[KINECT_NODES[NODE]].tolist()

            return kinect_skel
        else:
            return 0
    def get_skel_quat(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            # xyz = []
            quat = []
            for joint in body3d_joints:
                # xyz.append(joint.get_xyz())
                quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break

            # xyz = np.array(xyz)

            # np.save('xyz.npy', xyz)
            quat = np.array(quat) ## [32,4]
            # np.save('quat.npy', quat)
            # quat[:,1:] = -quat[:,1:]
            # print(quat.shape)
            quat = Quaternions(quat)

            quat[KINECT_NODES['PELVIS'] ] = quat[KINECT_NODES['PELVIS'] ]*-spineHipBasis
            # 
            # quat[KINECT_NODES['PELVIS'] ] = X_180_FLIP*quat[KINECT_NODES['PELVIS'] ]*-spineHipBasis
            

            # quat[KINECT_NODES['SPINE_NAVEL'] ] =X_180_FLIP*quat[KINECT_NODES['SPINE_NAVEL'] ] * -spineHipBasis 
            quat[KINECT_NODES['SPINE_NAVEL'] ] =quat[KINECT_NODES['SPINE_NAVEL'] ] * -spineHipBasis 
            quat[KINECT_NODES['SPINE_NAVEL'] ] = -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['SPINE_NAVEL'] ]


            quat[KINECT_NODES['SPINE_CHEST'] ] = X_180_FLIP*quat[KINECT_NODES['SPINE_CHEST'] ] * -spineHipBasis 
            # quat[KINECT_NODES['SPINE_CHEST'] ] = quat[KINECT_NODES['SPINE_CHEST'] ] * -spineHipBasis 
            quat[KINECT_NODES['SPINE_CHEST'] ] = -quat[KINECT_NODES['SPINE_NAVEL'] ] * quat[KINECT_NODES['SPINE_CHEST'] ]

            quat[KINECT_NODES['NECK'] ] = X_180_FLIP*quat[KINECT_NODES['NECK'] ]*-spineHipBasis
            quat[KINECT_NODES['NECK'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ]*quat[KINECT_NODES['NECK'] ]

            quat[KINECT_NODES['NOSE'] ] = X_180_FLIP*quat[KINECT_NODES['NOSE'] ]*-spineHipBasis
            # quat[KINECT_NODES['NOSE'] ] = -quat[KINECT_NODES['NECK'] ] *quat[KINECT_NODES['NOSE'] ]

            quat[KINECT_NODES['HIP_LEFT'] ] = quat[KINECT_NODES['HIP_LEFT'] ] * -hipleftBasis 
            # quat[KINECT_NODES['HIP_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['HIP_LEFT'] ] * -hipleftBasis 
            quat[KINECT_NODES['HIP_LEFT'] ] = -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['HIP_LEFT'] ]

            quat[KINECT_NODES['KNEE_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['KNEE_LEFT'] ] * -hipleftBasis 
            # quat[KINECT_NODES['KNEE_LEFT'] ] = quat[KINECT_NODES['KNEE_LEFT'] ] * -hipleftBasis 
            quat[KINECT_NODES['KNEE_LEFT'] ] = -quat[KINECT_NODES['HIP_LEFT'] ] * quat[KINECT_NODES['KNEE_LEFT'] ]

            quat[KINECT_NODES['HIP_RIGHT'] ] = quat[KINECT_NODES['HIP_RIGHT'] ] * -hiprightBasis 
            quat[KINECT_NODES['HIP_RIGHT'] ] = -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['HIP_RIGHT'] ]
            # quat[KINECT_NODES['HIP_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['HIP_RIGHT'] ] * -hiprightBasis 


            quat[KINECT_NODES['KNEE_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['KNEE_RIGHT'] ] * -hiprightBasis 
            # quat[KINECT_NODES['KNEE_RIGHT'] ] = quat[KINECT_NODES['KNEE_RIGHT'] ] * -hiprightBasis 
            quat[KINECT_NODES['KNEE_RIGHT'] ] = -quat[KINECT_NODES['HIP_RIGHT'] ]*quat[KINECT_NODES['KNEE_RIGHT'] ] 

            # quat[KINECT_NODES['SHOULDER_LEFT'] ] = quat[KINECT_NODES['SHOULDER_LEFT'] ] 
            quat[KINECT_NODES['SHOULDER_LEFT'] ] = quat[KINECT_NODES['SHOULDER_LEFT'] ] * -leftArmBasis 
            quat[KINECT_NODES['SHOULDER_LEFT'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ] * quat[KINECT_NODES['SHOULDER_LEFT'] ]

            quat[KINECT_NODES['ELBOW_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['ELBOW_LEFT'] ] * -leftArmBasis 
            quat[KINECT_NODES['ELBOW_LEFT'] ] = -quat[KINECT_NODES['SHOULDER_LEFT'] ] * quat[KINECT_NODES['ELBOW_LEFT'] ]
            # quat[KINECT_NODES['HIP_RIGHT'] ] =  -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['HIP_RIGHT'] ] 

            # quat[KINECT_NODES['SHOULDER_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['SHOULDER_RIGHT'] ] * -rightArmBasis 
            quat[KINECT_NODES['SHOULDER_RIGHT'] ] =quat[KINECT_NODES['SHOULDER_RIGHT'] ] * -rightArmBasis 
            quat[KINECT_NODES['SHOULDER_RIGHT'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ] * quat[KINECT_NODES['SHOULDER_RIGHT'] ]

            quat[KINECT_NODES['ELBOW_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['ELBOW_RIGHT'] ] * -rightArmBasis 
            quat[KINECT_NODES['ELBOW_RIGHT'] ] = -quat[KINECT_NODES['SHOULDER_RIGHT'] ]*quat[KINECT_NODES['ELBOW_RIGHT'] ]

            for NODE in KINECT_NODES:
                kinect_skel[NODE] = np.array(quat[KINECT_NODES[NODE] ]).squeeze().tolist()
            
            kinect_skel['basis'] = np.array(LookRotation(forward=xpositive, up=zpositive)).squeeze().tolist()
            return kinect_skel
        else:
            return '0'
        
    def stop_device(self):
        self.device.close()
    
if __name__ == '__main__':

    kin = KinectLib()
    # print(kin.get_skel_quat())
    print(kin.get_skel_pos())
    kin.device.close()