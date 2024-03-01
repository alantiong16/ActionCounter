from ultralytics import YOLO
import cv2
import os
import supervision as sv
import pyrealsense2 as rs
import numpy as np

import open3d as o3d
import imageio.v3 as iio
import numpy as np
import open3d as o3d
import time

import tensorflow as tf
import tensorflow_hub as hub
import sys

import time
import argparse


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth)
cfg = pipeline.start()
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() 
ppx, ppy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy

# print(intr)

a = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
# print(a.intrinsic_matrix)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
# pipeline.start(config)


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = device.first_depth_sensor()
# device.hardware_reset()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

align = rs.align(rs.stream.color)

threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.max_distance, 1)
def main():
    
    model = YOLO("yolov8n.pt") 
    cls_id = [0]
    
    model_select = 1
    args = get_args()
    if args.file is not None:
        cap_device = args.file
    keypoint_score_th = args.keypoint_score

    if model_select == 0:
        model_path = 'lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        input_size = 192
    elif model_select == 1:
        model_path = 'lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        input_size = 256
    elif model_select == 2:
        model_path = 'lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        input_size = 192
    elif model_select == 3:
        model_path = 'lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        input_size = 256
    elif model_select == 4:
        model_path = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
        input_size = 256
    else:
        sys.exit(
            "*** model_select {} is invalid value. Please use 0-3. ***".format(
                model_select))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    prev_frame_time = 0
    new_frame_time = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud Visualization", width=600, height=400)
    pcd = o3d.geometry.PointCloud()
    array =np.zeros((17, 3))
    leg_raise_count = 0
    squat_count = 0
    show_lr_count = False
    show_squat_count = False
    check1 = False
    check2 = False
    check1_right = False
    check2_right = False
    squat1 = False
    squat2 = False
    try:
        while True:

            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                    continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth_image = cv2.flip(depth_image, 1)
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            color_image = cv2.flip(color_image, 1)
            frame = cv2.cvtColor(color_image , cv2.COLOR_BGR2RGB)
            results = model.track(frame, persist=True , show=False, stream=True, agnostic_nms=True, tracker="bytetrack.yaml" , device=0)
            
            xyxys = []
            confidences = []


            for r in results:
                r = r[r.boxes.cls == 0 ]
                r = r[r.boxes.conf > 0.5]    

                for boxes in r.boxes.cpu().numpy():
                    for box in boxes:
                        xyxy = box.xyxy[0].astype(int)
                        conf = box.conf[0]
                        xyxys.append(xyxy)
                        confidences.append(conf)
            
            # with Timer('Detected Object Time'):
                if (len(confidences) > 0 ):

                    # with Timer('Object detection'):
                        highest_human_conf = confidences.index(max(confidences))
                        b = xyxys[highest_human_conf] 
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(frame, "person", (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,)
                        x_min, y_min, x_max, y_max = b #b[0], b[1], b[2], b[3]
                        # create empty image for the region of interest
                        height = y_max - y_min
                        width = x_max - x_min
                        roi = np.zeros((height, width, 3), dtype=np.float16)
                        roi_depth = np.zeros((height, width, 3), dtype=np.uint16)

                    # with Timer('Pixel to ROI'):
                        # for y in range(y_min, y_max): 
                        #     for x in range(x_min, x_max):
                        #         roi[y - y_min, x - x_min] = frame[y, x]
                        #         roi_depth[y - y_min, x - x_min] = depth_image[y, x]
                        # cv2.imshow("roi", roi)
                        # Create slices for the region of interest
                        roi_margin = 50  # You can change this value to control the margin size
                        x_min = max(0, b[0] - roi_margin)
                        y_min = max(0, b[1] - roi_margin)
                        x_max = min(frame.shape[1], b[2] + roi_margin)
                        y_max = min(frame.shape[0], b[3] + roi_margin)
                        roi = frame[y_min:y_max, x_min:x_max]
                        roi_depth = depth_image[y_min:y_max, x_min:x_max]
                        debug_image = roi
   
                        keypoints, scores = run_inference(
                            interpreter,
                            input_size,
                            roi,
                        )

                        new_frame_time =time.time()
                        fps =1/(new_frame_time-prev_frame_time)
                        prev_frame_time= new_frame_time

                        debug_image = draw_debug(
                            debug_image,
                            fps,
                            keypoint_score_th,
                            keypoints,
                            scores,
                        )
                        

                        # show the foreground mask in a window called "Foreground Mask"
                        
                        # thunder.main()
                        # break the loop if the 'q' key is pressed

                        # Set the threshold for keypoint scores
                        keypoint_score_th = 0.4
                        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(roi_depth, alpha=0.03), cv2.COLORMAP_JET)
                        # Iterate through the keypoints and scores
                        # for index, (keypoint, score) in enumerate(zip(keypoints, scores)):
                        #     # Check if the score is above the threshold
                        #     if score > keypoint_score_th:
                        #         # Retrieve the (x, y) coordinates of the keypoint
                        #         x, y = keypoint
                        #         # Draw a circle at the (x, y) coordinates on the depth image
                        #         cv2.circle(depth_colormap, (x, y), 6, (255, 255, 255), -1)
                        #         cv2.circle(depth_colormap, (x, y), 3, (0, 0, 0), -1)

                        #         # Connect the keypoints with lines
                        #         # Define the connections based on the indices of the keypoints
                                
                        #         for connection in connections:
                        #             index01, index02 = connection
                        #             if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
                        #                 point01 = keypoints[index01]
                        #                 point02 = keypoints[index02]
                        #                 # cv2.line(depth_colormap, point01, point02, (255, 255, 255), 4)
                        #                 # cv2.line(depth_colormap, point01, point02, (0, 0, 0), 2)

                        # Display the depth image with the connected keypoints
                        # cv2.imshow('Depth Image with Keypoints', depth_colormap)

                        with Timer('counting'):
                            # Convert depth image to point cloud
                            # points = []
                            # for v in range(roi_depth.shape[0]):
                            #     for u in range(roi_depth.shape[1]):
                            #         depth_values = roi_depth[v, u] * depth_scale
                            #         valid_depth_elements = depth_values[depth_values > 0]
                            #         for depth in valid_depth_elements:
                            #             x = (u - x_min) * depth / intr.fx
                            #             y = (v - y_min) * depth / intr.fy
                            #             z = depth
                            #             points.append([x, y, z])
                            # Assuming roi_depth is a numpy array
                            depth_values = roi_depth * depth_scale
                            # valid_indices = np.where(depth_values > 0)
                            # v_values = valid_indices[0] - y_min
                            # u_values = valid_indices[1] - x_min
                            # x_values = u_values * depth_values[valid_indices] / intr.fx
                            # y_values = v_values * depth_values[valid_indices] / intr.fy
                            # points = np.column_stack((x_values, y_values, depth_values[valid_indices]))
                            points_3d = []
                            # Counter for leg raises
                            for index, (keypoint, score) in enumerate(zip(keypoints, scores)):
                                x, y = keypoint
                                # Calculate 3D coordinates
                                if 0 <= x < depth_values.shape[1] and 0 <= y < depth_values.shape[0]:
                                    depth = depth_values[y, x]   # Access depth value at (x, y)
                                    x_3d = (x-intr.ppx) * depth / intr.fx
                                    y_3d = (y-intr.ppy) * depth / intr.fy
                                    if (depth!=0 or x_3d!=0 or y_3d!=0):
                                        points_3d.append([x_3d, y_3d, depth])
                                    else:
                                        points_3d.append(array[index])

                                    if index == 5:
                                        # print(f"Left Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        left_shoulder_x_3d = abs(x_3d)
                                        left_shoulder_y_3d = abs(y_3d)
                                        left_shoulder_depth = abs(depth)
                                    if index == 6:
                                        # print(f"Left Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        right_shoulder_x_3d = abs(x_3d)
                                        right_shoulder_y_3d = abs(y_3d)
                                        right_shoulder_depth = abs(depth)
                                    if index == 7:
                                        # print(f"Left Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        left_elbow_x_3d = abs(x_3d)
                                        left_elbow_y_3d = abs(y_3d)
                                        left_elbow_depth = abs(depth)
                                    if index == 8:
                                        # print(f"Left Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        right_elbow_x_3d = abs(x_3d)
                                        right_elbow_y_3d = abs(y_3d)
                                        right_elbow_depth = abs(depth)
                                       
                                    if index == 11:
                                        # print(f"Left Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        left_hips_x_3d = abs(x_3d)
                                        left_hips_y_3d = abs(y_3d)
                                        left_hips_depth = abs(depth) 
                                    if index == 12:
                                        # print(f"Right Hips 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")    
                                        right_hips_x_3d = abs(x_3d)
                                        right_hips_y_3d = abs(y_3d)
                                        right_hips_depth = abs(depth)        
                                    if index == 13:
                                        # print(f"Right Knee 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")
                                        right_knee_x_3d = abs(x_3d)
                                        right_knee_y_3d = abs(y_3d)
                                        right_knee_depth = abs(depth)
                                    if index == 14:
                                        # print(f"Left Knee 3D Coordinates: X={x_3d}, Y={y_3d}, Depth={depth}")
                                        left_knee_x_3d = abs(x_3d)
                                        left_knee_y_3d = abs(y_3d)
                                        left_knee_depth = abs(depth)  
                                    

                                        
                                        left_shoulder = np.array([left_shoulder_x_3d, left_shoulder_y_3d, left_shoulder_depth])
                                        left_hips = np.array([left_hips_x_3d, left_hips_y_3d, left_hips_depth])
                                        left_knee = np.array([left_knee_x_3d, left_knee_y_3d, left_knee_depth])
                                        left_elbow = np.array([left_elbow_x_3d, left_elbow_y_3d, left_elbow_depth])
                                        right_shoulder = np.array([right_shoulder_x_3d, right_shoulder_y_3d, right_shoulder_depth])
                                        right_hips = np.array([right_hips_x_3d, right_hips_y_3d, right_hips_depth])
                                        right_knee = np.array([right_knee_x_3d, right_knee_y_3d, right_knee_depth])
                                        right_elbow = np.array([right_elbow_x_3d, right_elbow_y_3d, right_elbow_depth])

                                        # # leg raise
                                        # # Calculate vectors between the joints
                                        # vector_shoulder_to_hips = left_hips - left_shoulder
                                        # vector_hips_to_knee = left_hips - left_knee

                                        # # Calculate dot product and magnitudes
                                        # dot_product = np.dot(vector_shoulder_to_hips, vector_hips_to_knee)
                                        # magnitude_shoulder_to_hips = np.linalg.norm(vector_shoulder_to_hips)
                                        # magnitude_hips_to_knee = np.linalg.norm(vector_hips_to_knee)
                                        # # Calculate the cosine of the angle between the vectors
                                        # cosine_angle = dot_product / (magnitude_shoulder_to_hips * magnitude_hips_to_knee)
                                        # # Calculate the angle in radians
                                        # angle_in_radians = np.arccos(cosine_angle)
                                        # # Convert radians to degrees
                                        # leg_angle_in_degrees = np.degrees(angle_in_radians)
                                        # if 120 <= leg_angle_in_degrees <= 140:
                                        #     check1=True
                                        # if 80 <= leg_angle_in_degrees <= 100 and check1==True:
                                        #     check2=True
                                        # if check1==True and check2==True:
                                        #     leg_raise_count+=1
                                        #     check1=False
                                        #     check2=False   
                                        
                                        # # Calculate vectors between the joints for the right side
                                        # vector_shoulder_to_hips_right =  right_shoulder - right_hips
                                        # vector_hips_to_knee_right = right_hips - right_knee

                                        # # Calculate dot product and magnitudes for the right side
                                        # dot_product_right = np.dot(vector_shoulder_to_hips_right, vector_hips_to_knee_right)
                                        # magnitude_shoulder_to_hips_right = np.linalg.norm(vector_shoulder_to_hips_right)
                                        # magnitude_hips_to_knee_right = np.linalg.norm(vector_hips_to_knee_right)

                                        # # Calculate the cosine of the angle between the vectors for the right side
                                        # cosine_angle_right = dot_product_right / (magnitude_shoulder_to_hips_right * magnitude_hips_to_knee_right)

                                        # # Calculate the angle in radians for the right side
                                        # angle_in_radians_right = np.arccos(cosine_angle_right)

                                        # # Convert radians to degrees for the right side
                                        # leg_angle_in_degrees_right = np.degrees(angle_in_radians_right)

                                        # if 120 <= leg_angle_in_degrees_right <= 140:
                                        #     check1_right = True

                                        # if 80 <= leg_angle_in_degrees_right <= 100 and check1_right == True:
                                        #     check2_right = True

                                        # if check1_right == True and check2_right == True:
                                        #     leg_raise_count += 1
                                        #     check1_right = False
                                        #     check2_right = False
                                        # print(f"Angle between left shoulder to left hips and left hips to left knee: {angle_in_degrees} degrees")    
                                        # if 70<=leg_angle_in_degrees<=100 or 70 <= leg_angle_in_degrees_right <= 100:
                                        #     check1=check1_right=True
                                        # if (0 <= leg_angle_in_degrees <= 10 or 0 <= leg_angle_in_degrees_right <= 10) and (check1 == True or check1_right == True):
                                        #     check2=check2_right=True
                                        # if check1==True and check2==True:
                                        #     show_lr_count = True
                                        #     leg_raise_count+=1
                                        #     check1=check2=check1_right=check2_right=False
                                        
                                        # # Squat Count  
                                        # if 120 <= leg_angle_in_degrees_right <= 140 and 120 <= leg_angle_in_degrees <= 140:
                                        #     squat1 = True
                                        # if 80 <= leg_angle_in_degrees_right <= 100 and 80 <= leg_angle_in_degrees <= 100 and squat1 == True:
                                        #     squat2 = True
                                        # if squat1 == True and squat2 == True:
                                        #     squat_count += 1
                                        #     squat1 = False
                                        #     squat2 = False
                                        
                                        # # Jumping Jack
                                        # vector_hips_to_shoulder = left_hips - left_shoulder
                                        # vector_shoulder_to_elbow = left_shoulder - left_elbow
                                        # vector_hips_to_shoulder_right = right_hips - right_shoulder
                                        # vector_shoulder_to_elbow_right = right_shoulder - right_elbow
                                        # vector_lhip_to_rhip = left_hips - right_hips
                                        # vector_rhip_to_rknee = right_hips - right_knee
                                        # vector_rhip_to_lhip = right_hips - left_hips
                                        # vector_lhip_to_lknee = left_hips - left_knee
                                        # # Calculate dot product and magnitudes
                                        # dot_product_jj = np.dot(vector_hips_to_shoulder, vector_shoulder_to_elbow)
                                        # magnitude_hips_to_shoulder = np.linalg.norm(vector_hips_to_shoulder)
                                        # magnitude_shoulder_to_elbow = np.linalg.norm(vector_shoulder_to_elbow)
                                        # # Calculate the cosine of the angle between the vectors
                                        # cosine_angle = dot_product_jj / (magnitude_hips_to_shoulder * magnitude_shoulder_to_elbow)
                                        # # Calculate the angle in radians
                                        # shoulder_angle_radians = np.arccos(cosine_angle)
                                        # # Convert radians to degrees
                                        # shoulder_angle_in_degree = np.degrees(shoulder_angle_radians)

                                        # # dot_product_jj_right = np.dot(vector_hips_to_shoulder_right, vector_shoulder_to_elbow_right)
                                        # # magnitude_hips_to_shoulder_right = np.linalg.norm(vector_hips_to_shoulder_right)
                                        # # magnitude_shoulder_to_elbow_right = np.linalg.norm(vector_shoulder_to_elbow_right)
                                        # # # Calculate the cosine of the angle between the vectors
                                        # # cosine_angle = dot_product_jj_right / (magnitude_hips_to_shoulder_right * magnitude_shoulder_to_elbow_right)
                                        # # # Calculate the angle in radians
                                        # # shoulder_angle_radians_right = np.arccos(cosine_angle)
                                        # # # Convert radians to degrees
                                        # shoulder_angle_in_degree_right = np.degrees(shoulder_angle_radians_right)

                                        # # Calculate dot product and magnitudes
                                        # dot_product_jj_rleg = np.dot(vector_lhip_to_rhip, vector_rhip_to_rknee)
                                        # magnitude_lhip_to_rhip = np.linalg.norm(vector_lhip_to_rhip)
                                        # magnitude_rhip_to_rknee = np.linalg.norm(vector_rhip_to_rknee)
                                        # # Calculate the cosine of the angle between the vectors
                                        # cosine_angle_rleg = dot_product_jj_rleg / (magnitude_lhip_to_rhip * magnitude_rhip_to_rknee)
                                        # # Calculate the angle in radians
                                        # shoulder_angle_radians_rleg = np.arccos(cosine_angle_rleg)
                                        # # Convert radians to degrees
                                        # shoulder_angle_in_degree_rleg = np.degrees(shoulder_angle_radians_rleg)

                                        # dot_product_jj_right_lleg = np.dot(vector_rhip_to_lhip, vector_lhip_to_lknee)
                                        # magnitude_hips_to_shoulder_right_lleg = np.linalg.norm(vector_rhip_to_lhip)
                                        # magnitude_shoulder_to_elbow_right_lleg = np.linalg.norm(vector_lhip_to_lknee)
                                        # # Calculate the cosine of the angle between the vectors
                                        # cosine_angle_lleg = dot_product_jj_right_lleg / (magnitude_hips_to_shoulder_right_lleg * magnitude_shoulder_to_elbow_right_lleg)
                                        # # Calculate the angle in radians
                                        # shoulder_angle_radians_lleg = np.arccos(cosine_angle_lleg)
                                        # # Convert radians to degrees
                                        # shoulder_angle_in_degree_lleg = np.degrees(shoulder_angle_radians_lleg)
                                        cv2.putText(debug_image, "1 : " + '{:}'.format(shoulder_angle_in_degree) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                                        cv2.putText(debug_image, "1 : " + '{:}'.format(shoulder_angle_in_degree) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        cv2.putText(debug_image, "2 : " + '{:}'.format(shoulder_angle_in_degree_right) , (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                                        cv2.putText(debug_image, "2 : " + '{:}'.format(shoulder_angle_in_degree_right) , (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        # cv2.putText(debug_image, "3 : " + '{:}'.format(shoulder_angle_in_degree_lleg) , (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                                        # cv2.putText(debug_image, "3 : " + '{:}'.format(shoulder_angle_in_degree_lleg) , (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        # cv2.putText(debug_image, "4 : " + '{:}'.format(shoulder_angle_in_degree_rleg) , (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        # cv2.putText(debug_image, "4 : " + '{:}'.format(shoulder_angle_in_degree_rleg) , (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                                        # if show_squat_count == True:
                                        cv2.putText(debug_image, "Squat : " + '{:}'.format(squat_count) , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                                        cv2.putText(debug_image, "Squat : " + '{:}'.format(squat_count) , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        # if show_lr_count == True:
                                        # cv2.putText(debug_image, "Leg Raise : " + '{:}'.format(leg_raise_count) , (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                                        # cv2.putText(debug_image, "Leg Raise : " + '{:}'.format(leg_raise_count) , (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                        
                            # cv2.putText(debug_image, "distance : " + '{:}'.format(distance) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                            # cv2.putText(debug_image, "distance : " + '{:}'.format(distance) , (10, 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_knee_x_3d : " + '{:}'.format(left_knee_x_3d) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_knee_x_3d : " + '{:}'.format(left_knee_x_3d) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_hips_y_3d : " + '{:}'.format(left_hips_y_3d) , (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_hips_y_3d : " + '{:}'.format(left_hips_y_3d) , (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_knee_y_3d : " + '{:}'.format(left_knee_y_3d) , (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
                            # cv2.putText(debug_image, "left_knee_y_3d : " + '{:}'.format(left_knee_y_3d) , (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                            
                            # Convert the list of 3D points to a NumPy array
                            points_3d = np.array(points_3d, dtype=np.float32)
                            # print("last array:", array)
                            array = points_3d
                            # Initialize lists to store valid connections
                            print(array)
                            
                    # with Timer('pointcloud'):
                        # Create a point cloud object
                        # lines = []
                        # for connection in connections:
                        #     start_idx, end_idx = connection
                        #     if 0 <= start_idx < len(points_3d) and 0 <= end_idx < len(points_3d):

                        #         start_point = points_3d[start_idx]
                        #         end_point = points_3d[end_idx]
                        #         lines.append([start_point, end_point])

                    # Define the indices of the keypoints for eye, nose, and right eye

                        
                        # with Timer('draw point cloud'):# Initialize lists to store valid connections
                        valid_connections = []
                        for connection in connections:
                            index01, index02 = connection
                            # Check if both keypoints meet the score threshold and have valid 3D positions
                            valid_connections.append(connection)
                            # print(len(valid_connections))
                            
                            # Create a line set using Open3D
                        # print(valid_connections)
                        vis.clear_geometries()
                        pcd.points = o3d.utility.Vector3dVector(points_3d)
                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(points_3d)
                        line_set.lines = o3d.utility.Vector2iVector(valid_connections)  
                        transform_matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                        pcd.transform(transform_matrix)
                        line_set.transform(transform_matrix)
                        distance_threshold = 3.5
                        points_np = np.asarray(pcd.points)
                        distances = np.linalg.norm(points_np, axis=1)
                        within_threshold_mask = distances <= distance_threshold
                        filtered_pcd = pcd.select_by_index(np.where(within_threshold_mask)[0])
                        draw_xyz_grid(vis, scale=4, step=0.1, reference_points=points_3d)
                        vis.add_geometry(filtered_pcd)
                        vis.add_geometry(line_set)  # Add the line set
                        vis.update_geometry(filtered_pcd)
                        vis.update_geometry(line_set)
                        vis.poll_events()
                        vis.update_renderer()

                        # Measure Distance 
                        # X_depth = int((x_max/2) * aligned_depth_frame.width / color_frame.width)
                        # Y_depth = int((y_max/2) *  aligned_depth_frame.height / color_frame.height)  
                        # depth_value = aligned_depth_frame.get_distance(X_depth, Y_depth)
                        # depth_in_meters = depth_value
                        # z_max = np.max(points_3d [:, 2])
                        # z_min = np.min(points_3d [:, 1])
                        # cv2.putText(debug_image, "height : " + '{:.2f}'.format(depth_in_meters) + "metre", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
                        # cv2.putText(debug_image, "height : " + '{:.2f}'.format(depth_in_meters) + "metre", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)  
                        cv2.imshow('Foreground Mask', debug_image)
                        # cv2.imshow("output", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()

# Define a function to calculate 3D Euclidean distance


def draw_xyz_grid(vis, scale=1, step=0.1, reference_points=None):
    # Create a point cloud for the grid lines
    grid_points = []

    # Create the colors for each axis
    x_color = [1, 0, 0]
    y_color = [0, 1, 0]
    z_color = [0, 0, 1]

    num_points = int((scale / step) + 1)
    num_points2 = int(2 * (scale / step) + 2)
    uniform_x_color = [x_color] * num_points
    uniform_y_color = [y_color] * num_points
    uniform_z_color = [z_color] * num_points

    # Create grid lines along the X-axis
    for x in np.arange(0, scale + step, step):
        grid_points.append([x, 0, 0])

    # Create grid lines along the Y-axis
    for y in np.arange(0, scale + step, step):
        grid_points.append([0, y, 0])

    # Create grid lines along the Z-axis
    for z in np.arange(0, scale + step, step):
        grid_points.append([0, 0, z])

    if reference_points is not None:
        translation = np.mean(reference_points, axis=0)
    else:
        translation = [0, 0, 0]  # Default to the origin if no reference points are provided

    grid_points = np.array(grid_points) + translation

    grid_pcd = o3d.geometry.PointCloud()
    grid_pcd.points = o3d.utility.Vector3dVector(grid_points)

    # Set colors for each axis
    grid_pcd.paint_uniform_color(x_color)
    grid_pcd.colors[:num_points] = o3d.utility.Vector3dVector(uniform_x_color)
    grid_pcd.colors[num_points:num_points2] = o3d.utility.Vector3dVector(uniform_y_color)
    grid_pcd.colors[num_points2:] = o3d.utility.Vector3dVector(uniform_z_color)

    vis.add_geometry(grid_pcd)

# Example usage:
# draw_xyz_grid(vis, scale=1, step=0.1, reference_points=None)


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = image
    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    # Line：鼻 → 左目
    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右目
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左目 → 左耳
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右目 → 右耳
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # # Line：鼻 → 左肩
    # index01, index02 = 0, 5
    # if scores[index01] > keypoint_score_th and scores[
    #         index02] > keypoint_score_th:
    #     point01 = keypoints[index01]
    #     point02 = keypoints[index02]
    #     cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
    #     cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # # Line：鼻 → 右肩
    # index01, index02 = 0, 6
    # if scores[index01] > keypoint_score_th and scores[
    #         index02] > keypoint_score_th:
    #     point01 = keypoints[index01]
    #     point02 = keypoints[index02]
    #     cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
    #     cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 右肩
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左肘
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肘 → 左手首
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右肘
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肘 → 右手首
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 右股関節
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左股関節
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 左ひざ
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左ひざ → 左足首
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右股関節
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右股関節 → 右ひざ
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右ひざ → 右足首
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[
            index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv2.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv2.line(debug_image, point01, point02, (0, 0, 0), 2)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv2.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv2.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # 処理時間
    # cv2.putText(debug_image,
    #            "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "ms",
    #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
    #            cv2.LINE_AA)
    # cv2.putText(debug_image,
    #            "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "ms",
    #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
    #            cv2.LINE_AA)

    return debug_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)

    args = parser.parse_args()

    return args

def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.uint8)  # uint8へキャスト

    # 推論
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、スコア取り出し
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]
        
        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores

if __name__ == '__main__':
    main()