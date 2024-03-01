from plot3dUtils import Open3dVisualizer, Open3dMeshVisualizer
import sys
sys.path.append(r'C:\Users\cic\Documents\SMPL\smplpytorch')
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch
import numpy as np
import time
import open3d as o3d

def compare_pc(verts, joints):
    verts_pc = o3d.geometry.PointCloud()
    joints_pc = o3d.geometry.PointCloud()
    verts_pc.points = o3d.utility.Vector3dVector(verts)
    verts_pc.paint_uniform_color([1, 0.706, 0])
    joints_pc.points = o3d.utility.Vector3dVector(joints)
    joints_pc.paint_uniform_color([0, 0.706, 1])
    # pcd_combined_down.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([verts_pc, joints_pc])

def o3d_meshviewer(model, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(model)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    device = 'cuda'
    model = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root=r'C:\Users\cic\Documents\SMPL\modelsv1.0')
    model.to(device).type(torch.float64)
    faces = model.th_faces
    pose_size = 72
    beta_size = 10
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    beta = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    pose = torch.from_numpy(pose).type(torch.float64).to(device).unsqueeze(0)
    beta = torch.from_numpy(beta).type(torch.float64).to(device).unsqueeze(0)
    trans = torch.from_numpy(trans).type(torch.float64).to(device).unsqueeze(0)

    with torch.no_grad():
        verts, joints = model(pose, beta, trans)
    compare_pc(verts.cpu().numpy()[0], joints.cpu().numpy()[0])
    # o3d_meshviewer(verts.cpu().numpy()[0], faces=faces.cpu().numpy())
    