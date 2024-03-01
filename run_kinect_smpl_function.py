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
from KinectLib_filter import KinectLib
if __name__ == "__main__":

	kin = KinectLib()
	kin.start_service() 

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
	verts_offset = torch.load('verts_offset.pt', map_location='cuda')
	smpl.th_v_template = smpl.th_v_template + verts_offset
	# star = STAR(gender='male')
	betas = np.array([
				np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
						2.20098416, 0.26102114, -3.07428093, 0.55708514,
						-3.94442258, -2.88552087])])
	num_betas=10

	poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
	trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
	betas = torch.cuda.FloatTensor(np.zeros_like(betas))

	

	try:
		while True:

			skel_json = kin.get_item()
			
			if not skel_json == '0': 
				
				q_left_shoulder = skel_json['SHOULDER_LEFT']
				# print()
				quat = torch.from_numpy(np.array(q_left_shoulder))
				# print(quat.shape)
				aa = transforms.quaternion_to_axis_angle(quat)
				# left_elbow = aa[6,:]
				poses = poses.reshape(1, -1, 3)

				# to_display = ['Hips','spine','spine1', 'LeftUpLeg','LeftLeg', 'RightUpLeg', 'RightLeg','LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm']
				# to_display = ['Hips', 'RightArm','LeftArm', 'LeftForeArm', 'RightForeArm', 'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'spine', 'spine1', 'Neck', 'Head']
				# for node in to_display:
				# 	star_idx = star_nodes[node]
				# 	poses[0, star_idx , :]= aa[kinect_nodes[kinect_to_star[star_idx]],:]


				# poses[0, star_nodes['RightArm'], :]= aa[kinect_nodes['SHOULDER_RIGHT'],:]
				poses[0, star_nodes['LeftArm'], :]= aa
				# poses[0, star_nodes['RightFoot'], :]= aa[kinect_nodes['ANKLE_RIGHT'],:]
				# poses[0, star_nodes['RightLeg'], :]= aa[kinect_nodes['KNEE_RIGHT'],:]

				poses = poses.reshape(1,-1)
				# poZ = vp.encode(poses[:,3:66]).mean
				# recon_pose = vp.decode(poZ)['pose_body'].contiguous().view(-1, 63)
				# poses[:,3:66] = recon_pose
				with torch.no_grad():
					model, joints = smpl(poses, betas, trans)
				# model = star.forward(poses, betas,trans)
				# print(smpl.th_faces.dtype)
				o3dmesh_viz.update(model[0].cpu().numpy(), location=xyz[0]/1000 * np.array([1,1,-1]), faces=smpl.th_faces.cpu().numpy())
				# o3d_viz.update(xyz)
			# break
	except KeyboardInterrupt:
  		kin.stop_service()
	# device.close()
		
