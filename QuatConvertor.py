import cv2
import sys
import numpy as np
# sys.path.append('C:\\Users\\cic\\Documents\\pyKinectAzure')
# from pykinect_azure.pykinect_azure import *
import pykinect_azure as pykinect
from plot3dUtils import Open3dVisualizer, Open3dMeshVisualizer
# print (pykinect.__file__)
sys.path.append(r'../bvh_utils')
sys.path.append(r'../STAR')
sys.path.append(r'C:/Users/cic/Documents/human_body_prior/src')
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.body_model.body_model import BodyModel
from star.pytorch.star import STAR
from Quaternions_old import Quaternions
from pytorch3d import transforms
import torch
import json
from QuaternionLookRotation import LookRotation
star_nodes = {
            'Hips': 0,
            'LeftUpLeg': 1,
            'RightUpLeg': 2,
            'spine': 3,
            'LeftLeg': 4,
            'RightLeg': 5,
            'spine1': 6,
            'LeftFoot': 7,
            'RightFoot': 8,
            'spine2': 9,
            'LeftToeBase': 10,
            'RightToeBase': 11,
            'Neck': 12,
            'shoulderleft': 13,
            'RightShoulder': 14,
            'Head': 15,
            'LeftArm': 16,
            'RightArm': 17,
            'LeftForeArm': 18,
            'RightForeArm': 19,
            'LeftHand': 20,
            'RightHand': 21,
	    	'LeftPalm' : 22,
	    	'RightPalm' : 23,
		    
        }

kinect_to_star = ['PELVIS',
	       		'HIP_LEFT',
				'HIP_RIGHT',
				'SPINE_NAVEL',
				'KNEE_LEFT',
				'KNEE_RIGHT',
				'SPINE_CHEST',
				'ANKLE_LEFT',
				'ANKLE_RIGHT',
				None,
				'FOOT_LEFT',
				'FOOT_RIGHT',
				'NECK',
				None,
				None,
				'NOSE',
				'SHOULDER_LEFT',
				'SHOULDER_RIGHT',
				'ELBOW_LEFT',
				'ELBOW_RIGHT',
				'WRIST_LEFT',
				'WRIST_RIGHT',
				'HANDTIP_LEFT',
				'HANDTIP_RIGHT'
				]

# kinect_nodes = {0:'PELVIS',
# 				1:'SPINE_NAVAL',
# 				2:'SPINE_CHEST',
# 				3:'NECK',
# 				4:'CLAVICLE_LEFT',
# 				5:'SHOULDER_LEFT',
# 				6:'ELBOW_LEFT',
# 				7:'WRIST_LEFT',
# 				8:'HAND_LEFT',
# 				9:'HANDTIP_LEFT',
# 				10:'THUMB_LEFT',
# 				11:'CLAVICLE_RIGHT',
# 				12:'SHOULDER_RIGHT',
# 				13:'ELBOW_RIGHT',
# 				14:'WRIST_RIGHT',
# 				15:'HAND_RIGHT',
# 				16:'HANDTIP_RIGHT',
# 				17:'THUMB_RIGHT',
# 				18:'HIP_LEFT',
# 				19:'KNEE_LEFT',
# 				20:'ANKLE_LEFT',
# 				21:'FOOT_LEFT',
# 				22:'HIP_RIGHT',
# 				23:'KNEE_RIGHT',
# 				24:'ANKLE_RIGHT',
# 				25:'FOOT_RIGHT',
# 				26:'HEAD',
# 				27:'NOSE',
# 				28:'EYE_LEFT',
# 				29:'EAR_LEFT',
# 				30:'EYE_RIGHT',
# 				31:'EAR_RIGHT'}
kinect_nodes = {'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3, 'CLAVICLE_LEFT': 4, 'SHOULDER_LEFT': 5, 'ELBOW_LEFT': 6, 'WRIST_LEFT': 7, 'HAND_LEFT': 8, 'HANDTIP_LEFT': 9, 'THUMB_LEFT': 10, 'CLAVICLE_RIGHT': 11, 'SHOULDER_RIGHT': 12, 'ELBOW_RIGHT': 13, 'WRIST_RIGHT': 14, 'HAND_RIGHT': 15, 'HANDTIP_RIGHT': 16, 'THUMB_RIGHT': 17, 'HIP_LEFT': 18, 'KNEE_LEFT': 19, 'ANKLE_LEFT': 20, 'FOOT_LEFT': 21, 'HIP_RIGHT': 22, 'KNEE_RIGHT': 23, 'ANKLE_RIGHT': 24, 'FOOT_RIGHT': 25, 'HEAD': 26, 'NOSE': 27, 'EYE_LEFT': 28, 'EAR_LEFT': 29, 'EYE_RIGHT': 30, 'EAR_RIGHT': 31}
kinect_to_star_node = []
star_node = []
for idx, _node in enumerate(kinect_to_star):
	if _node == None:
		continue
	star_node.append(idx)
	kinect_to_star_node.append(kinect_nodes[_node])

zpositive = np.array([0,0,-1])
xpositive = np.array([1,0,0])
ypositive = np.array([0,1,0])
X_180_FLIP = Quaternions(np.array([0,1,0,0]))
Y_180_FLIP = Quaternions(np.array([0,0,1,0]))
Z_180_FLIP = Quaternions(np.array([0,0,0,1]))

hipleftBasis = LookRotation(xpositive, -zpositive)
spineHipBasis = LookRotation(xpositive, -zpositive)
hiprightBasis = LookRotation(xpositive, zpositive)

leftArmBasis = LookRotation(ypositive, -zpositive)

rightArmBasis = LookRotation(-xpositive, zpositive)

leftHandBasis = LookRotation(-zpositive, -ypositive)
rightHandBasis = Quaternions(np.array([1,0,0,0]))
leftFootBasis = LookRotation(xpositive, ypositive)
rightFootBasis = LookRotation(xpositive, -ypositive)

basisJointMap = {}
basisJointMap['PELVIS'] = spineHipBasis
basisJointMap['SPINE_NAVEL'] = spineHipBasis
basisJointMap['SPINE_CHEST'] = spineHipBasis
basisJointMap['NECK'] = spineHipBasis
basisJointMap['ClavicleLeft'] = leftArmBasis
basisJointMap['SHOULDER_LEFT'] = leftArmBasis
basisJointMap['ELBOW_LEFT'] = leftArmBasis
basisJointMap['WRIST_LEFT'] = leftHandBasis
basisJointMap['HandLeft'] = leftHandBasis
basisJointMap['HANDTIP_LEFT'] = leftHandBasis
basisJointMap['ThumbLeft'] = leftArmBasis
basisJointMap['ClavicleRight'] = rightArmBasis
basisJointMap['SHOULDER_RIGHT'] = rightArmBasis
basisJointMap['ELBOW_RIGHT'] = rightArmBasis
basisJointMap['WRIST_RIGHT'] = rightHandBasis
basisJointMap['HandRight'] = rightHandBasis
basisJointMap['HANDTIP_RIGHT'] = rightHandBasis
basisJointMap['ThumbRight'] = rightArmBasis
basisJointMap['HIP_LEFT'] = hipleftBasis
basisJointMap['KNEE_LEFT'] = hipleftBasis
basisJointMap['ANKLE_LEFT'] = hipleftBasis
basisJointMap['FOOT_LEFT'] = leftFootBasis
basisJointMap['HIP_RIGHT'] = hiprightBasis
basisJointMap['KNEE_RIGHT'] = hiprightBasis
basisJointMap['ANKLE_RIGHT'] = hiprightBasis
basisJointMap['FOOT_RIGHT'] = rightFootBasis
basisJointMap['Head'] = spineHipBasis
basisJointMap['NOSE'] = spineHipBasis
basisJointMap['EyeLeft'] = spineHipBasis
basisJointMap['EarLeft'] = spineHipBasis
basisJointMap['EyeRight'] = spineHipBasis
basisJointMap['EarRight'] = spineHipBasis
print(basisJointMap)


def quat_conv(quat):
	quat = Quaternions(quat)
	# print(quat.shape)
	# shoulderright = Quaternions(quat[kinect_nodes['SHOULDER_RIGHT'],:])
	# shoulderleft = Quaternions(quat[kinect_nodes['SHOULDER_LEFT'],:])
	# print(shoulderleft)



	# rotated_shoulderright = Y_180_FLIP*shoulderright * -rightArmBasis
	# _node = star_nodes['Hips']
	# k_node = kinect_nodes[_node]

	rotated_pelvis = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['PELVIS']]
	w,x,y,z = rotated_pelvis.qs[0]
	quat[kinect_nodes['PELVIS'] ] = Quaternions(np.array([w,z,x,y]))

	rotated_hipleft = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['HIP_LEFT']]
	w,x,y,z = rotated_hipleft.qs[0]
	quat[kinect_nodes['HIP_LEFT'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['PELVIS'] ]
	
	rotated_kneeleft = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['KNEE_LEFT']]
	w,x,y,z = rotated_kneeleft.qs[0]
	quat[kinect_nodes['KNEE_LEFT'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['HIP_LEFT'] ]
	# quat[kinect_nodes['KNEE_LEFT'] ] = rotated_kneeleft *-quat[kinect_nodes['HIP_LEFT'] ]

	rotated_hipright = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['HIP_RIGHT']]
	w,x,y,z = rotated_hipright.qs[0]
	quat[kinect_nodes['HIP_RIGHT'] ] = X_180_FLIP * Quaternions(np.array([w,-z,x,y])) *-quat[kinect_nodes['PELVIS'] ]
	quat[kinect_nodes['HIP_RIGHT'] ] = Quaternions(quat[kinect_nodes['HIP_RIGHT'] ].qs * np.array([1,-1,1,1]))
	
	rotated_kneeright = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['KNEE_RIGHT']]
	w,x,y,z = rotated_kneeright.qs[0]
	# quat[kinect_nodes['KNEE_RIGHT'] ] = X_180_FLIP * Quaternions(np.array([w,-z,x,y])) 
	quat[kinect_nodes['KNEE_RIGHT'] ] = X_180_FLIP * Quaternions(np.array([w,-z,x,y])) *-quat[kinect_nodes['HIP_RIGHT'] ]
	quat[kinect_nodes['KNEE_RIGHT'] ] = Quaternions(quat[kinect_nodes['KNEE_RIGHT'] ].qs * np.array([1,1,1,1]))


	rotated_spinenavel = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['SPINE_NAVEL']]
	w,x,y,z = rotated_spinenavel.qs[0]
	quat[kinect_nodes['SPINE_NAVEL'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['PELVIS'] ]

	rotated_spinechest = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['SPINE_CHEST']]
	w,x,y,z = rotated_spinechest.qs[0]
	quat[kinect_nodes['SPINE_CHEST'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['SPINE_NAVEL'] ]

	rotated_neck = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['NECK']]
	w,x,y,z = rotated_neck.qs[0]
	quat[kinect_nodes['NECK'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['SPINE_CHEST'] ]

	rotated_nose = Quaternions.from_euler(np.array([np.pi/2,0,np.pi/2])) * quat[kinect_nodes['NOSE']]
	w,x,y,z = rotated_nose.qs[0]
	quat[kinect_nodes['NOSE'] ] = Quaternions(np.array([w,z,x,y])) *-quat[kinect_nodes['NECK'] ]




	rotated_shoulderright = Quaternions.from_euler(np.array([-np.pi/2,0,0])) * quat[kinect_nodes['SHOULDER_RIGHT']]
	w,x,y,z = rotated_shoulderright.qs[0]
	quat[kinect_nodes['SHOULDER_RIGHT'] ] = Quaternions(np.array([w,x,z,-y])) *-quat[kinect_nodes['SPINE_CHEST'] ]


	rotated_elbowright = Quaternions.from_euler(np.array([-np.pi/2,0,0])) * quat[kinect_nodes['ELBOW_RIGHT']]
	w,x,y,z = rotated_elbowright.qs[0]
	quat[kinect_nodes['ELBOW_RIGHT'] ] = Quaternions(np.array([w,x,z,-y])) * -quat[kinect_nodes['SHOULDER_RIGHT'] ]


	rotated_shoulderleft = Quaternions.from_euler(np.array([np.pi/2,0,0])) * quat[kinect_nodes['SHOULDER_LEFT']]
	w,x,y,z = rotated_shoulderleft.qs[0]
	quat[kinect_nodes['SHOULDER_LEFT']] = Quaternions(np.array([w,x,-z,y]))*-quat[kinect_nodes['SPINE_CHEST'] ]


	# rotated_elbowleft = quat[kinect_nodes['ELBOW_LEFT']]
	# w,x,y,z = rotated_elbowleft.qs[0]
	# quat[kinect_nodes['ELBOW_LEFT']] = Quaternions(np.array([w,x,-z,y]))
	# print(rotated_elbowleft)
	rotated_elbowleft = Quaternions.from_euler(np.array([np.pi/2,0,0])) * quat[kinect_nodes['ELBOW_LEFT']]
	w,x,y,z = rotated_elbowleft.qs[0]
	quat[kinect_nodes['ELBOW_LEFT']] = Quaternions(np.array([w,x,-z,y])) * -quat[kinect_nodes['SHOULDER_LEFT']]


	quat = torch.from_numpy(np.array(quat.normalized()))
	# print(quat.shape)
	aa = transforms.quaternion_to_axis_angle(quat)
	# left_elbow = aa[6,:]
	poses = torch.zeros(quat.shape[0],24,3)
	# poses = poses.reshape(1, -1, 3)

	to_display = ['Hips', 'RightArm','LeftArm', 'LeftForeArm', 'RightForeArm', 'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'spine', 'spine1', 'Neck', 'Head']
	for node in to_display:
		star_idx = star_nodes[node]
		poses[:, star_idx , :]= aa[:,kinect_nodes[kinect_to_star[star_idx]],:]
	# poses[0, star_nodes['RightArm'], :]= aa[kinect_nodes['SHOULDER_RIGHT'],:]
	# poses[0, star_nodes['LeftArm'], :]= aa[kinect_nodes['SHOULDER_LEFT'],:]
	# poses[0, star_nodes['RightFoot'], :]= aa[kinect_nodes['ANKLE_RIGHT'],:]
	# poses[0, star_nodes['RightLeg'], :]= aa[kinect_nodes['KNEE_RIGHT'],:]

	poses = poses.reshape(quat.shape[0],-1)
	return poses
