import open3d as o3d
import numpy as np
from QuaternionLookRotation import LookRotation
import sys
sys.path.append(r'../bvh_utils')
from Quaternions_old import Quaternions
import copy

if __name__ == '__main__':
    zpositive = np.array([0,0,1])
    xpositive = np.array([1,0,0])
    ypositive = np.array([0,1,0])
    X_180_FLIP = Quaternions(np.array([0,1,0,0]))
    Y_180_FLIP = Quaternions(np.array([0,0,1,0]))
    Z_180_FLIP = Quaternions(np.array([0,0,0,1]))


    hipleftBasis = LookRotation(forward=xpositive, up=zpositive)
    # hipleftBasis =  -hipleftBasis*hipleftBasis
    # print(hipleftBasis)
    # print(f'basis {hipleftBasis.euler() * 180 / np.pi}')
    # qr = Quaternions.from_euler(np.array([-np.pi/4,0,0]))
    # hipleftBasis =  qr*-hipleftBasis*hipleftBasis
    # print(r)
    # angles = hipleftBasis.euler()
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_mv = copy.deepcopy(coor).translate((1, 1, 0), relative=False)

    R = coor.get_rotation_matrix_from_quaternion(np.array(hipleftBasis).transpose(1, 0))
    # R = coor.get_rotation_matrix_from_euler(np.array([np.pi/4,0,0]))



    coor.rotate(R, center=(0, 0, 0))

    bbox = coor.get_axis_aligned_bounding_box()
    bbox.color = np.array([0, 0.706, 1])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(model)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    o3d.visualization.draw_geometries([coor,bbox,mesh_mv])