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

sys.path.append(r'C:\Users\cic\Documents\SMPL\smplpytorch')
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import os.path as osp
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
# print(star_node)
# print(kinect_to_star_node)
if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	#print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker()
	o3dmesh_viz = Open3dMeshVisualizer()
	# o3d_viz = Open3dVisualizer()
	# cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)
	batch_size = 1

	vposer_expr_dir = osp.join(r'C:\Users\cic\Documents\human_body_prior\support_data\dowloads\vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
	bm_fname =  osp.join(r'C:\Users\cic\Documents\human_body_prior\support_data\dowloads\models\smplx\neutral\model.npz')#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
	comp_device = 'cuda'
	expr_dir = osp.join(r'C:\Users\cic\Documents\human_body_prior\support_data\dowloads\vposer_v2_05')
	vp, ps = load_model(expr_dir, model_code=VPoser,
								remove_words_in_model_weights='vp_model.',
								disable_grad=True)
	
	# for name, p in vp.named_parameters():
	# 	# print(name, p.requires_grad)
	# 	if "encoder_net" in name:
	# 		p.requires_grad = True
			# print(name, p.requires_grad)
	vp.to('cuda')
	vp.eval()
	# smpl = SMPLModel(model_path=os.path.join(r'C:\Users\cic\Documents\SMPL\modelsv1.0\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'), device='cuda')
	smpl = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root=r'C:\Users\cic\Documents\SMPL\modelsv1.0')
	smpl.to('cuda')
	smpl.eval()
	
	# star = STAR(gender='male')
	betas = np.array([
				np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
						2.20098416, 0.26102114, -3.07428093, 0.55708514,
						-3.94442258, -2.88552087])])
	num_betas=10

	poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
	trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
	betas = torch.cuda.FloatTensor(np.zeros_like(betas))

	zpositive = np.array([0,0,1])
	xpositive = np.array([1,0,0])
	ypositive = np.array([0,1,0])
	X_180_FLIP = Quaternions(np.array([0,1,0,0]))
	Y_180_FLIP = Quaternions(np.array([0,0,1,0]))
	Z_180_FLIP = Quaternions(np.array([0,0,0,1]))

	# hipleftBasis = LookRotation(forward=ypositive, up=xpositive)
	# spineHipBasis = LookRotation(forward=ypositive, up=xpositive)
	hipleftBasis = LookRotation(forward=xpositive, up=zpositive)
	spineHipBasis = LookRotation(forward=xpositive, up=zpositive)
	hiprightBasis = LookRotation(xpositive, -zpositive)

	leftArmBasis = LookRotation(-ypositive, zpositive)
	
	rightArmBasis = LookRotation(ypositive, -zpositive)

	leftHandBasis = LookRotation(zpositive, -ypositive)
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


	while True:

		# Get capture
		capture = device.update()

		# Get body tracker frame
		body_frame = bodyTracker.update()

		if body_frame.get_num_bodies():
			body3d = body_frame.get_body(0)
			body3d_joints = body3d.joints
			xyz = []
			quat = []
			for joint in body3d_joints:
				xyz.append(joint.get_xyz())
				quat.append(joint.get_quaternion())
				# print(xyz)
			# print(body3d.joints[0].get_xyz())
			# break

			xyz = np.array(xyz)
			
			# np.save('xyz.npy', xyz)
			quat = np.array(quat) ## [32,4]
			# np.save('quat.npy', quat)
			# quat[:,1:] = -quat[:,1:]
			# print(quat.shape)
			quat = Quaternions(quat)
			
			# shoulderright = Quaternions(quat[kinect_nodes['SHOULDER_RIGHT'],:])
			# shoulderleft = Quaternions(quat[kinect_nodes['SHOULDER_LEFT'],:])
			# print(shoulderleft)



			# rotated_shoulderright = Y_180_FLIP*shoulderright * -rightArmBasis
			# _node = star_nodes['Hips']
			# k_node = kinect_nodes[_node]
			quat[kinect_nodes['PELVIS'] ] = X_180_FLIP*quat[kinect_nodes['PELVIS'] ]*-spineHipBasis

			quat[kinect_nodes['SPINE_NAVEL'] ] =X_180_FLIP*quat[kinect_nodes['SPINE_NAVEL'] ] * -spineHipBasis 
			quat[kinect_nodes['SPINE_NAVEL'] ] = -quat[kinect_nodes['PELVIS'] ] * quat[kinect_nodes['SPINE_NAVEL'] ]
			quat[kinect_nodes['SPINE_CHEST'] ] = X_180_FLIP*quat[kinect_nodes['SPINE_CHEST'] ] * -spineHipBasis 
			quat[kinect_nodes['SPINE_CHEST'] ] = -quat[kinect_nodes['SPINE_NAVEL'] ] * quat[kinect_nodes['SPINE_CHEST'] ]

			quat[kinect_nodes['NECK'] ] = X_180_FLIP*quat[kinect_nodes['NECK'] ]*-spineHipBasis
			quat[kinect_nodes['NECK'] ] = -quat[kinect_nodes['SPINE_CHEST'] ]*quat[kinect_nodes['NECK'] ]

			quat[kinect_nodes['NOSE'] ] = X_180_FLIP*quat[kinect_nodes['NOSE'] ]*-spineHipBasis
			# quat[kinect_nodes['NOSE'] ] = -quat[kinect_nodes['NECK'] ] *quat[kinect_nodes['NOSE'] ]

			quat[kinect_nodes['HIP_LEFT'] ] = X_180_FLIP*quat[kinect_nodes['HIP_LEFT'] ] * -hipleftBasis 
			# quat[kinect_nodes['HIP_LEFT'] ] = -quat[kinect_nodes['PELVIS'] ] * quat[kinect_nodes['HIP_LEFT'] ]
			quat[kinect_nodes['KNEE_LEFT'] ] = X_180_FLIP*quat[kinect_nodes['KNEE_LEFT'] ] * -hipleftBasis 
			quat[kinect_nodes['KNEE_LEFT'] ] = -quat[kinect_nodes['HIP_LEFT'] ] * quat[kinect_nodes['KNEE_LEFT'] ]

			quat[kinect_nodes['HIP_RIGHT'] ] = X_180_FLIP*quat[kinect_nodes['HIP_RIGHT'] ] * -hiprightBasis 
			quat[kinect_nodes['KNEE_RIGHT'] ] = X_180_FLIP*quat[kinect_nodes['KNEE_RIGHT'] ] * -hiprightBasis 
			quat[kinect_nodes['KNEE_RIGHT'] ] = -quat[kinect_nodes['HIP_RIGHT'] ]*quat[kinect_nodes['KNEE_RIGHT'] ] 

			quat[kinect_nodes['SHOULDER_LEFT'] ] = X_180_FLIP*quat[kinect_nodes['SHOULDER_LEFT'] ] * -leftArmBasis 
			quat[kinect_nodes['SHOULDER_LEFT'] ] = -quat[kinect_nodes['SPINE_CHEST'] ] * quat[kinect_nodes['SHOULDER_LEFT'] ]

			quat[kinect_nodes['ELBOW_LEFT'] ] = X_180_FLIP*quat[kinect_nodes['ELBOW_LEFT'] ] * -leftArmBasis 
			quat[kinect_nodes['ELBOW_LEFT'] ] = -quat[kinect_nodes['SHOULDER_LEFT'] ] * quat[kinect_nodes['ELBOW_LEFT'] ]
			# quat[kinect_nodes['HIP_RIGHT'] ] =  -quat[kinect_nodes['PELVIS'] ] * quat[kinect_nodes['HIP_RIGHT'] ] 
			
			quat[kinect_nodes['SHOULDER_RIGHT'] ] = X_180_FLIP*quat[kinect_nodes['SHOULDER_RIGHT'] ] * -rightArmBasis 
			quat[kinect_nodes['SHOULDER_RIGHT'] ] = -quat[kinect_nodes['SPINE_CHEST'] ] * quat[kinect_nodes['SHOULDER_RIGHT'] ]

			quat[kinect_nodes['ELBOW_RIGHT'] ] = X_180_FLIP*quat[kinect_nodes['ELBOW_RIGHT'] ] * -rightArmBasis 
			quat[kinect_nodes['ELBOW_RIGHT'] ] = -quat[kinect_nodes['SHOULDER_RIGHT'] ]*quat[kinect_nodes['ELBOW_RIGHT'] ]  


			quat = torch.from_numpy(np.array(quat.normalized()))
			# print(quat.shape)
			aa = transforms.quaternion_to_axis_angle(quat)
			# left_elbow = aa[6,:]
			poses = poses.reshape(1, -1, 3)

			# to_display = ['Hips','spine','spine1', 'LeftUpLeg','LeftLeg', 'RightUpLeg', 'RightLeg','LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm']
			to_display = ['Hips', 'RightArm','LeftArm', 'LeftForeArm', 'RightForeArm', 'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'spine', 'spine1', 'Neck', 'Head']
			for node in to_display:
				star_idx = star_nodes[node]
				poses[0, star_idx , :]= aa[kinect_nodes[kinect_to_star[star_idx]],:]
			# poses[0, star_nodes['RightArm'], :]= aa[kinect_nodes['SHOULDER_RIGHT'],:]
			# poses[0, star_nodes['LeftArm'], :]= aa[kinect_nodes['SHOULDER_LEFT'],:]
			# poses[0, star_nodes['RightFoot'], :]= aa[kinect_nodes['ANKLE_RIGHT'],:]
			# poses[0, star_nodes['RightLeg'], :]= aa[kinect_nodes['KNEE_RIGHT'],:]

			poses = poses.reshape(1,-1)
			# poZ = vp.encode(poses[:,3:66]).mean
			# recon_pose = vp.decode(poZ)['pose_body'].contiguous().view(-1, 63)
			# poses[:,3:66] = recon_pose
			model, joints = smpl(poses, betas, trans)
			# model = star.forward(poses, betas,trans)
			# print(smpl.th_faces.dtype)
			o3dmesh_viz.update(model[0].cpu().numpy(), location=xyz[0]/1000 * np.array([1,1,-1]), faces=smpl.th_faces.cpu().numpy())
			# o3d_viz.update(xyz)
		# break
		
