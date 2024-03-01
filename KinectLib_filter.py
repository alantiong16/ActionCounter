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
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import time
from datetime import timedelta

import queue
import threading

SAMPLE_DATA = {'timestamp': 6121766711, 'PELVIS': [-0.7612315490841866, 0.34352294355630875, 0.5131271407008171, 0.19803788512945175], 'SPINE_NAVEL': [0.99872390705115, 0.050502073470497164, -1.6811355210699652e-08, -1.286666639899181e-08], 'SPINE_CHEST': [-0.8328857535571366, 0.2539290458803262, 0.4814469024239117, 0.10015081467539398], 'NECK': [0.9934161541569485, 0.1060840797018598, 0.01766441705677085, -0.03947673216019844], 'CLAVICLE_LEFT': [-0.49992260336875916, 0.6801138520240784, -0.2461431473493576, -0.47637802362442017], 'SHOULDER_LEFT': [-0.8024493694508766, -0.09105738033992211, 0.3591305894244684, 0.4677700006942126], 'ELBOW_LEFT': [-0.46842665896544333, -0.23108190546277502, 0.8423648761312716, 0.13266102090243165], 'WRIST_LEFT': [-0.18756254017353058, -0.07182865589857101, -0.46306371688842773, 0.8632687330245972], 'HAND_LEFT': [0.11550650745630264, 0.08904319256544113, -0.46006375551223755, 0.875825822353363], 'HANDTIP_LEFT': [0.11550650745630264, 0.08904319256544113, -0.46006375551223755, 0.875825822353363], 'THUMB_LEFT': [0.05061645060777664, 0.06168483570218086, 0.3175470530986786, 0.9448792338371277], 'CLAVICLE_RIGHT': [0.7908836603164673, 0.500150203704834, 0.2763615548610687, -0.21903693675994873], 'SHOULDER_RIGHT': [0.8755992506999692, 0.13159928657396633, 0.345183510679001, 0.31121670549490554], 'ELBOW_RIGHT': [-0.8850700988891519, 0.2811509046535739, -0.02866147330634857, 0.36984258552808513], 'WRIST_RIGHT': [0.8126028180122375, -0.5630929470062256, -0.1447591632604599, 0.040593087673187256], 'HAND_RIGHT': [0.7927872538566589, -0.5797775387763977, -0.043437667191028595, 0.18291941285133362], 'HANDTIP_RIGHT': [0.7927872538566589, -0.5797775387763977, -0.043437667191028595, 0.18291941285133362], 'THUMB_RIGHT': [0.9834131598472595, 0.12397512048482895, -0.1306418627500534, 0.021482745185494423], 'HIP_LEFT': [-0.03486523473225178, 0.02927011445795835, 0.45837385645265244, 0.8875928437550198], 'KNEE_LEFT': [-0.8083602076396574, 0.20961628868830867, -0.42036958934383123, 0.3548297148816598], 'ANKLE_LEFT': [-0.15405157208442688, 0.8128763437271118, -0.5154367685317993, 0.2232154905796051], 'FOOT_LEFT': [0.271918922662735, -0.2136579304933548, 0.9370421171188354, -0.04860459268093109], 'HIP_RIGHT': [-0.3666899062227458, 0.4772531830240041, -0.6110905713867396, -0.5141364329028875], 'KNEE_RIGHT': [0.8223802669206225, -0.5689382092956853, -6.902162344840646e-05, 9.448612230472664e-05], 'ANKLE_RIGHT': [0.19236783683300018, 0.5519759654998779, 0.8109722137451172, 0.02532375603914261], 'FOOT_RIGHT': [0.11979219317436218, 0.9638869166374207, 0.18636131286621094, 0.147787943482399], 'HEAD': [-0.1468021422624588, 0.7352051734924316, -0.5898047089576721, 0.30008816719055176], 'NOSE': [-0.856362813158795, 0.22508219374104885, 0.4572961935982989, -0.08283042821008663], 'EYE_LEFT': [0.31599918007850647, -0.10281363129615784, 0.9369235038757324, -0.10838956385850906], 'EAR_LEFT': [0.30008816719055176, 0.5898047089576721, 0.7352051734924316, 0.1468021422624588], 'EYE_RIGHT': [0.31599918007850647, -0.10281363129615784, 0.9369235038757324, -0.10838956385850906], 'EAR_RIGHT': [-0.1468021422624588, 0.7352051734924316, -0.5898047089576721, 0.30008816719055176], 'basis': [0.7071067811865476, 0.7071067811865475, 0.0, 0.0]}
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
        self.init_output = False
        self.q_buffer = []
        self.service_start = False
        self.q_list = queue.Queue()
        self.init_output= False

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
        
    def get_item(self):

        return self.q_list.get()
        pass
    
    def stop_service(self):
        self.service_start = False
        pass
    def start_service(self):
        self.service_start = True
        t = threading.Thread(target=self.start_capture)
        t.start()

    def start_capture(self):
        while self.service_start:
            inter_rot, q_json = self.gen_slerp()
            
            for rot in inter_rot:
                q_json['SHOULDER_LEFT'] =  rot
                self.q_list.put(q_json)
            # print(self.q_list.qsize())
        pass
    def get_skel_quat_filter_mock(self):
        time.sleep(0.0333)
        return SAMPLE_DATA
    
    # @classmethod
    def gen_slerp(self):
        # q1_time = time.perf_counter()
        # self.init_output= False
        win_size = 0.2
        if not self.init_output:
            self.q1 = self.get_skel_quat_filter()
            if self.q1 == '0':
                return '0'
            self.init_output = True
        
        time.sleep(win_size)
        self.q2 = self.get_skel_quat_filter() 
        if self.q2 == '0':
            return '0'
        key_rots = R.concatenate([R.from_quat(np.array(self.q1['SHOULDER_LEFT'])),
                                R.from_quat(np.array(self.q2['SHOULDER_LEFT']))])
        key_times = [0, win_size]
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, win_size, int(win_size*30))
        interp_rots = slerp(times)

        self.q1 = self.q2
        # self.init_buffer()
        # while len(self.q_buffer) < self.win_size:
        #     q = self.get_skel_quat_filter()
        #     if not q == '0':
        #         self.q_buffer.append(q['SHOULDER_LEFT'])

        # key_rots = [R.from_quat(np.array(self.q_buffer[0])),
        #             R.from_quat(np.array(self.q_buffer[-1]))]
        # key_rots = R.concatenate(key_rots)
        # # print(key_rots)
        # key_times = [0, 1]
        # slerp = Slerp(key_times, key_rots)
        # times = [x * 0.1 for x in range(0,10)]
        # interp_rots = slerp(times)
        # # q_slerp = Quaternions.slerp(q1, q2, 10)
                        
        # # print(interp_rots)
        # imu_data = interp_rots.as_euler('xyz', degrees=False)[0]

        return interp_rots.as_quat(), self.q2
        pass

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
            offsets = np.load('tposev1.npy')
            # print(offsets.shape)
            norm_offsets = offsets - offsets[0]

            # norm_offsets[6] =  # left elbow
            # norm_offsets[7] =  # left hand

            norm_offsets[13,1] = norm_offsets[6,1]  # right elbow
            norm_offsets[14,1] = norm_offsets[7,1] # right hand


            xyz = X_180_FLIP.transforms() @ np.array(xyz).T
            xyz = xyz.T.squeeze()
            root = xyz[0].copy()
            # xyz =xyz - xyz[0]
            
            xyz = xyz - norm_offsets

            # quat[KINECT_NODES['PELVIS'] ] = quat[KINECT_NODES['PELVIS'] ]*-LookRotation(forward=xpositive, up=-zpositive)
            quat = Quaternions(body3d_joints[0].get_quaternion())*-LookRotation(forward=xpositive, up=-zpositive)
            # print(quat.shape)
            euler = quat.euler() * 180 / np.pi
            # print(euler.shape)
            # root[2] < -1000 and 
            if not (root[0] < 550 and root[0] > -550 and (root[2] > -2200)):
                print('outside bound')
                return 0
            
            print(root)

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
    
    def get_skel_quat_filter(self):
        capture = self.device.update()

		# Get body tracker frame
        body_frame = self.bodyTracker.update()

        kinect_skel = {'timestamp': body_frame.get_device_timestamp_usec()}
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
    
    

class GraphIMU():
    def __init__(self, imu):
        
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.x_vals = []
        self.y_vals = [[],[],[]]
        self.imu = imu
        self.fig, self.axs = plt.subplots(1,3)
        self.lines = []

        for axis in self.axs:
        
            axis.set_ylim(-1.5, 1.5)
            axis.set_xlim(0, 300)
            axis.grid()
            line, = axis.plot([], [], lw = 3)
            self.lines.append(line)

        self.axs[0].set_title('AX',fontsize = 14)
        self.axs[1].set_title('AY',fontsize = 14)
        self.axs[2].set_title('AZ',fontsize = 14)


    def stop_stream(self):
        self.stop = True
        self.imu.stop_save()
    def start_stream(self):
        # Create two subplots in row 1 and column 1, 2
        # plt.gcf().subplots(1, 2)
        # print(self.imu.client)
        # print('strating stream')
        
        # while not self.stop:
        anim = FuncAnimation(self.fig, self.animate,
                            init_func = self.init,
                            # frames = 300,
                            interval = 30,
                            blit = True)
        # self.root.after(10, self.start_stream)
        # plt.pause(0.001)
            # await asyncio.sleep(0)
        # anim.event_source.start()
        plt.show()
        print('have quitted toplevel')

    def init(self):
        self.win_size = 10
        self.q_buffer = []
        for line in self.lines:
            line.set_data([], [])
        return self.lines
    
    def animate(self,i):
        # print(f'Running: {loop.is_running()} Close: {loop.is_closed()}')
        # print(i)
        
        # print(self.imu.q.qsize())
        if len(self.y_vals[0]) > 300:
            
            self.x_vals.pop(0)
            [_.pop(0) for _ in self.y_vals]
            # y_vals.pop(0)
            # y_vals2.pop(0)
        imu_data = kin.get_skel_quat_filter()
        # print(imu_data)
        if (not imu_data == '0'):
            imu_data = imu_data['SHOULDER_LEFT']
            # self.q_buffer.append(np.array(imu_data).copy())

            # if len(self.q_buffer) < self.win_size:
            #     print('buffer not full')
                
            # else:
            q = Quaternions(np.array(imu_data))
            imu_data = np.array(q.euler())[0]
            print(imu_data)
            for idx, data_idx in enumerate(range(0,3)):
            # for idx, data_idx in enumerate(range(19,22)):
            # for idx, data_idx in enumerate(range(22,25)):
            # for idx, data_idx in enumerate(range(6,9)):
                self.y_vals[idx].append(imu_data[data_idx])
            
            self.x_vals.append(i)
            
            if i > 300:
                for axis in self.axs:
                    axis.set_xlim(left=i-300, right=i)
            
            for idx, line in enumerate(self.lines):
                line.set_data(self.x_vals, self.y_vals[idx])
                # lines[1].set_data(x_vals, y_vals2)
        else:
            
            print('no skeleton found')
        return self.lines

class GraphIMU2():
    def __init__(self, imu):
        
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.x_vals = []
        self.y_vals = [[],[],[]]
        self.imu = imu
        self.fig, self.axs = plt.subplots(1,3)
        self.lines = []

        for axis in self.axs:
        
            axis.set_ylim(-1.5, 1.5)
            axis.set_xlim(0, 300)
            axis.grid()
            line, = axis.plot([], [], lw = 3)
            self.lines.append(line)

        self.axs[0].set_title('AX',fontsize = 14)
        self.axs[1].set_title('AY',fontsize = 14)
        self.axs[2].set_title('AZ',fontsize = 14)


    def stop_stream(self):
        self.stop = True
        self.imu.stop_save()
    def start_stream(self):
        # Create two subplots in row 1 and column 1, 2
        # plt.gcf().subplots(1, 2)
        # print(self.imu.client)
        # print('strating stream')
        
        # while not self.stop:
        anim = FuncAnimation(self.fig, self.animate,
                            init_func = self.init,
                            # frames = 300,
                            interval = 30,
                            blit = True)
        # self.root.after(10, self.start_stream)
        # plt.pause(0.001)
            # await asyncio.sleep(0)
        # anim.event_source.start()
        plt.show()
        print('have quitted toplevel')

    def init(self):
        self.win_size = 10
        self.q_buffer = []
        for line in self.lines:
            line.set_data([], [])
        return self.lines
    
    def animate(self,i):
        # print(f'Running: {loop.is_running()} Close: {loop.is_closed()}')
        # print(i)
        
        # print(self.imu.q.qsize())
        if len(self.y_vals[0]) > 300:
            
            self.x_vals.pop(0)
            [_.pop(0) for _ in self.y_vals]
            # y_vals.pop(0)
            # y_vals2.pop(0)
        imu_data = kin.get_skel_quat_filter()
        # print(imu_data)
        if (not imu_data == '0'):
            imu_data = imu_data['SHOULDER_LEFT']
            self.q_buffer.append(np.array(imu_data).copy())
            
            if len(self.q_buffer) < self.win_size:
                print('buffer not full')
                
            else:

                # q1 = Quaternions(np.array(self.q_buffer[0]))
                # q2 = Quaternions(np.array(self.q_buffer[-1]))

                key_rots = [R.from_quat(np.array(self.q_buffer[0])),
                            R.from_quat(np.array(self.q_buffer[-1]))]
                key_rots = R.concatenate(key_rots)
                # print(key_rots)
                key_times = [0, 1]
                slerp = Slerp(key_times, key_rots)
                times = [x * 0.1 for x in range(0,10)]
                interp_rots = slerp(times)
                # q_slerp = Quaternions.slerp(q1, q2, 10)
                
                # print(interp_rots)
                imu_data = interp_rots.as_euler('xyz', degrees=False)[0]
                print(f'frame: {i} data: {imu_data}')
                for idx, data_idx in enumerate(range(0,3)):
                # for idx, data_idx in enumerate(range(19,22)):
                # for idx, data_idx in enumerate(range(22,25)):
                # for idx, data_idx in enumerate(range(6,9)):
                    self.y_vals[idx].append(imu_data[data_idx])
                
                self.x_vals.append(i)
                
                if i > 300:
                    for axis in self.axs:
                        axis.set_xlim(left=i-300, right=i)
                
                for idx, line in enumerate(self.lines):
                    line.set_data(self.x_vals, self.y_vals[idx])
                    # lines[1].set_data(x_vals, y_vals2)
                self.q_buffer.pop(0)
        else:
            
            print('no skeleton found')
        return self.lines
    

if __name__ == '__main__':

    kin = KinectLib()
    kin.start_service()
    try:
        while True:
            print(kin.get_item())
            time.sleep(0.03)
    except KeyboardInterrupt:
        kin.stop_service()
    # graph = GraphIMU(kin)
    # graph.start_stream()
    # print(kin.get_skel_quat_filter())
    
    # q_buffer= []
    # win_size = 10
    # while True:
    #     quat = kin.get_skel_quat_filter()
    #     imu_data = imu_data['SHOULDER_LEFT']
    #     q_buffer.append(np.array(imu_data).copy())
    #     if len(q_buffer) < win_size:
    #         print('buffer not full')
            
    #     else:
    #         key_rots = R.concatenate(key_rots)

    #     # print(quat)
    #     print((quat['SHOULDER_LEFT']))
    # kin.device.close()