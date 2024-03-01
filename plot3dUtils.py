import numpy as np
import cv2
import open3d as o3d

class Open3dVisualizer():

	def __init__(self):

		self.point_cloud = o3d.geometry.PointCloud()
		self.o3d_started = False

		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window(window_name='Point')

	def __call__(self, points_3d, rgb_image=None):

		self.update(points_3d, rgb_image)

	def update(self, points_3d, rgb_image=None):

		# Add values to vectors
		self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
		if rgb_image is not None:
			colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
			self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

		self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.point_cloud)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.point_cloud)

		self.vis.poll_events()
		self.vis.update_renderer()

class Open3dMeshVisualizer():

	def __init__(self):

		self.mesh = o3d.geometry.TriangleMesh()
		self.o3d_started = False

		self.vis = o3d.visualization.Visualizer()
		self.coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
		self.coor.scale(0.2, center=self.coor.get_center())
		self.vis.create_window(window_name='Mesh')

	def __call__(self, model, rgb_image=None):

		self.update(model, rgb_image)

	def update(self, model, rgb_image=None, location=None, faces=None):

		# Add values to vectors
		if faces is None:
			try:
				face = model.f
			except:
				face = model.faces
		else:
			face = faces

		if location is not None:
			offset = location
		else:
			offset = 0
		
		self.mesh.vertices = o3d.utility.Vector3dVector(model+offset)
		self.mesh.triangles = o3d.utility.Vector3iVector(face)
		# try:
		# 	self.mesh.vertices = o3d.utility.Vector3dVector(model[0].cpu().numpy()+offset)
		# except:
		# 	self.mesh.vertices = o3d.utility.Vector3dVector(model.v[0].detach().cpu().numpy()+offset)
			# print(type(face))
		# try:
		# 	self.mesh.triangles = o3d.utility.Vector3iVector(face.detach().cpu().numpy())
		# except:
		# 	self.mesh.triangles = o3d.utility.Vector3iVector(face)
		self.mesh.compute_vertex_normals()
		self.mesh.compute_triangle_normals()

		# self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
		# if rgb_image is not None:
		# 	colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
		# 	self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

		# self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.mesh)
			self.vis.add_geometry(self.coor)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.mesh)

		self.vis.poll_events()
		self.vis.update_renderer()

class Visualizer():

	def __init__(self):

		self.mesh = o3d.geometry.TriangleMesh()
		self.o3d_started = False

		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window(window_name='Mesh')

	def __call__(self, model, rgb_image=None):

		self.update(model, rgb_image)

	def update(self, model, rgb_image=None):

		# Add values to vectors
		face = model.f
		try:
			self.mesh.vertices = o3d.utility.Vector3dVector(model[0].cpu().numpy())
		except:
			self.mesh.vertices = o3d.utility.Vector3dVector(model.v[0].detach().cpu().numpy())
			# print(type(face))
		self.mesh.triangles = o3d.utility.Vector3iVector(face.detach().cpu().numpy())
		self.mesh.compute_vertex_normals()
		self.mesh.compute_triangle_normals()

		# self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
		# if rgb_image is not None:
		# 	colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
		# 	self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

		# self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.mesh)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.mesh)

		self.vis.poll_events()
		self.vis.update_renderer()