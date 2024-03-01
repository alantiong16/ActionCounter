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
from scipy.spatial.transform import Rotation as R

KINECT_NODES = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}

zpositive = np.array([0,0,1]) # forward
xpositive = np.array([1,0,0]) # right
ypositive = np.array([0,1,0]) # up

X_90_FLIP = Quaternions.from_euler(np.array([np.pi/2,0,0]))
Y_90_FLIP = Quaternions.from_euler(np.array([0,np.pi/2,0]))
Z_90_FLIP = Quaternions.from_euler(np.array([0,0,np.pi/2]))

X_180_FLIP = Quaternions(np.array([0,1,0,0]))
Y_180_FLIP = Quaternions(np.array([0,0,1,0]))
Z_180_FLIP = Quaternions(np.array([0,0,0,1]))

hipleftBasis = LookRotation(forward=-xpositive, up=zpositive)
spineHipBasis = LookRotation(forward=xpositive, up=zpositive)
hiprightBasis = LookRotation(-xpositive, -zpositive)

leftArmBasis = LookRotation(-ypositive, zpositive)
# leftArmBasis = LookRotation(-ypositive, zpositive)
# leftArmBasis = LookRotation(-xpositive, -zpositive)
# leftArmBasis = LookRotation(ypositive, -zpositive)

rightArmBasis = LookRotation(ypositive, -zpositive)

leftHandBasis = LookRotation(zpositive, -ypositive)
rightHandBasis = Quaternions(np.array([1,0,0,0]))
leftFootBasis = LookRotation(xpositive, ypositive)
rightFootBasis = LookRotation(xpositive, -ypositive)

ori_basis = Quaternions(np.array([0.78,-0.6124,-0.04934, -0.07698]))
ori_euler_angle = np.rad2deg(np.array(ori_basis.euler()))
# print(ori_euler_angle)
class KinectLib:
    def __init__(self) -> None:
        
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
       
        self.device = pykinect.start_device(config=device_config)

        self.bodyTracker = pykinect.start_body_tracker()

    def get_skel_tpose(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            # print(body3d_joints)
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                # quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break
            # offset_root = xyz[0]
            # xyz = np.array(xyz) - offset_root
            # xyz = xyz @ X_180_FLIP.transforms()[0] + offset_root
            # print(np.array(xyz).shape)
            xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            xyz = xyz.T
            
            # json_string=''
            # for NODE in KINECT_NODES:
            #     rng = xyz[KINECT_NODES[NODE]].squeeze()
            #     # print(rng.shape)
            #     json_string+=f'{NODE}:{rng[0]},{rng[1]},{rng[2]}|'

            return xyz
        else:
            return 0

    def get_skel_pos_offset_str(self):
        
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            # print(body3d_joints)
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                # quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break
            # offset_root = xyz[0]
            # xyz = np.array(xyz) - offset_root
            # xyz = xyz @ X_180_FLIP.transforms()[0] + offset_root
            # print(np.array(xyz).shape)
            offsets = np.load('tpose.npy')
            print(offsets.shape)
            norm_offsets = offsets - offsets[0]

            xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            xyz = xyz.T.squeeze()
            root = xyz[0].copy()
            # xyz =xyz 
            # - xyz[0]
            xyz = xyz - norm_offsets

            l_shoulder = xyz[5]
            l_elbow = xyz[6]
            l_wrist = xyz[7]

            v1 = l_elbow - l_shoulder
            v2 = l_wrist - l_elbow
            # v1 = l_shoulder - l_elbow
            # v2 = l_elbow - l_wrist

            # q_l_elbow = Quaternions.between(v1, v2)
            # _rot_q = R.from_quat(np.array(q_l_elbow))
            # .inv()
            # inv_euler = _rot_q.as_euler('XYZ', True).squeeze()

            # quat = []
            # for joint in body3d_joints:
            #     # xyz.append(joint.get_xyz())
            #     quat.append(joint.get_quaternion())
            #     # print(xyz)
            # # print(body3d.joints[0].get_xyz())
            # # break

            # # xyz = np.array(xyz)

            # # np.save('xyz.npy', xyz)
            # quat = np.array(quat) ## [32,4]
            # # np.save('quat.npy', quat)
            # # quat[:,1:] = -quat[:,1:]
            # # print(quat.shape)
            # quat = Quaternions(quat)

            # quat[KINECT_NODES['PELVIS'] ] = quat[KINECT_NODES['PELVIS'] ]*-LookRotation(forward=xpositive, up=-zpositive)
            quat = Quaternions(body3d_joints[0].get_quaternion())*-LookRotation(forward=xpositive, up=-zpositive)
            # print(quat.shape)
            euler = quat.euler() * 180 / np.pi
            # print(euler.shape)

            json_string=''
            for NODE in KINECT_NODES:
                rng = xyz[KINECT_NODES[NODE]]
                if NODE == 'PELVIS':
                    json_string+=f'{NODE}:{root[0]},{root[1]},{root[2]}|'
                else:
                    
                    # print(rng.shape)
                    json_string+=f'{NODE}:{rng[0]},{rng[1]},{rng[2]}|'
                    # kinect_skel[NODE] = xyz[KINECT_NODES[NODE]].tolist()
            
            # if inv_euler[2] < 0:
            #     inv_euler[2] = inv_euler[2] +180
            # else:
            #     inv_euler[2] =inv_euler[2]-180 
            # print(inv_euler)
            json_string+=f'PELVIS_ROT:{euler[0,0]},{euler[0,1]},{euler[0,2]}'
            return json_string
        else:
            return 0
        
    def get_skel_pos_str(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            # print(body3d_joints)
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                # quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break
            # offset_root = xyz[0]
            # xyz = np.array(xyz) - offset_root
            # xyz = xyz @ X_180_FLIP.transforms()[0] + offset_root
            # print(np.array(xyz).shape)
            xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            xyz = xyz.T.squeeze()
            # xyz =xyz - xyz[0]
            json_string=''
            for NODE in KINECT_NODES:
                rng = xyz[KINECT_NODES[NODE]]
                # print(rng.shape)
                json_string+=f'{NODE}:{rng[0]},{rng[1]},{rng[2]}|'
                # kinect_skel[NODE] = xyz[KINECT_NODES[NODE]].tolist()

            return json_string
        else:
            return 0
        
    def get_skel_pos(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {}
        if body_frame.get_num_bodies():
            body3d = body_frame.get_body(0)
            body3d_joints = body3d.joints
            # print(body3d_joints)
            xyz = []
            # quat = []
            for joint in body3d_joints:
                xyz.append(joint.get_xyz())
                # quat.append(joint.get_quaternion())
                # print(xyz)
            # print(body3d.joints[0].get_xyz())
            # break
            # offset_root = xyz[0]
            # xyz = np.array(xyz) - offset_root
            # xyz = xyz @ X_180_FLIP.transforms()[0] + offset_root
            # print(np.array(xyz).shape)
            xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            xyz = xyz.T
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

            quat[KINECT_NODES['PELVIS'] ] = X_180_FLIP*quat[KINECT_NODES['PELVIS'] ]*-spineHipBasis
            # quat[KINECT_NODES['PELVIS'] ] = quat[KINECT_NODES['PELVIS'] ]*-spineHipBasis

            quat[KINECT_NODES['SPINE_NAVEL'] ] =X_180_FLIP*quat[KINECT_NODES['SPINE_NAVEL'] ] * -spineHipBasis 
            quat[KINECT_NODES['SPINE_NAVEL'] ] = -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['SPINE_NAVEL'] ]
            quat[KINECT_NODES['SPINE_CHEST'] ] = X_180_FLIP*quat[KINECT_NODES['SPINE_CHEST'] ] * -spineHipBasis 
            quat[KINECT_NODES['SPINE_CHEST'] ] = -quat[KINECT_NODES['SPINE_NAVEL'] ] * quat[KINECT_NODES['SPINE_CHEST'] ]

            quat[KINECT_NODES['NECK'] ] = X_180_FLIP*quat[KINECT_NODES['NECK'] ]*-spineHipBasis
            quat[KINECT_NODES['NECK'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ]*quat[KINECT_NODES['NECK'] ]

            quat[KINECT_NODES['NOSE'] ] = X_180_FLIP*quat[KINECT_NODES['NOSE'] ]*-spineHipBasis
            quat[KINECT_NODES['NOSE'] ] = -quat[KINECT_NODES['NECK'] ] *quat[KINECT_NODES['NOSE'] ]

            quat[KINECT_NODES['HIP_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['HIP_LEFT'] ] * -hipleftBasis 
            quat[KINECT_NODES['HIP_LEFT'] ] = -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['HIP_LEFT'] ]

            quat[KINECT_NODES['KNEE_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['KNEE_LEFT'] ] * -hipleftBasis 
            quat[KINECT_NODES['KNEE_LEFT'] ] = -quat[KINECT_NODES['HIP_LEFT'] ] * quat[KINECT_NODES['KNEE_LEFT'] ]

            quat[KINECT_NODES['HIP_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['HIP_RIGHT'] ] * -hiprightBasis 
            quat[KINECT_NODES['KNEE_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['KNEE_RIGHT'] ] * -hiprightBasis 
            quat[KINECT_NODES['KNEE_RIGHT'] ] = -quat[KINECT_NODES['HIP_RIGHT'] ]*quat[KINECT_NODES['KNEE_RIGHT'] ] 

            quat[KINECT_NODES['SHOULDER_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['SHOULDER_LEFT'] ] *-(  leftArmBasis)
            # * 
            # quat[KINECT_NODES['SHOULDER_LEFT'] ] = quat[KINECT_NODES['SHOULDER_LEFT'] ] * -leftArmBasis 
            quat[KINECT_NODES['SHOULDER_LEFT'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ] * quat[KINECT_NODES['SHOULDER_LEFT'] ]

            quat[KINECT_NODES['ELBOW_LEFT'] ] = X_180_FLIP*quat[KINECT_NODES['ELBOW_LEFT'] ] * -leftArmBasis 
            quat[KINECT_NODES['ELBOW_LEFT'] ] = -quat[KINECT_NODES['SHOULDER_LEFT'] ] * quat[KINECT_NODES['ELBOW_LEFT'] ]
            # quat[KINECT_NODES['HIP_RIGHT'] ] =  -quat[KINECT_NODES['PELVIS'] ] * quat[KINECT_NODES['HIP_RIGHT'] ] 

            quat[KINECT_NODES['SHOULDER_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['SHOULDER_RIGHT'] ] * -rightArmBasis 
            quat[KINECT_NODES['SHOULDER_RIGHT'] ] = -quat[KINECT_NODES['SPINE_CHEST'] ] * quat[KINECT_NODES['SHOULDER_RIGHT'] ]

            quat[KINECT_NODES['ELBOW_RIGHT'] ] = X_180_FLIP*quat[KINECT_NODES['ELBOW_RIGHT'] ] * -rightArmBasis 
            quat[KINECT_NODES['ELBOW_RIGHT'] ] = -quat[KINECT_NODES['SHOULDER_RIGHT'] ]*quat[KINECT_NODES['ELBOW_RIGHT'] ]    


            for NODE in KINECT_NODES:
                kinect_skel[NODE] = np.array(quat[KINECT_NODES[NODE] ]).squeeze().tolist()
            
            kinect_skel['basis'] = np.array(leftArmBasis).squeeze().tolist()
            return kinect_skel
        else:
            return '0'
        
    def stop_device(self):
        self.device.close()
    
if __name__ == '__main__':

    kin = KinectLib()
    # print(kin.get_skel_quat())
    # print(kin.get_skel_pos())
    print(kin.get_skel_pos())
    kin.device.close()